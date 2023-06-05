// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_buffers.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

InsertBuffers::InsertBuffers(int32_t buffer_allocation_rank)
    : Pass(), m_buffer_allocation_rank(buffer_allocation_rank) {}

LinearIR::constExprIt InsertBuffers::insertion_position(const LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager,
                                                        const ExpressionPtr& up_expr, const ExpressionPtr& down_expr) {
    const auto up_loops = up_expr->get_loop_ids();
    const auto down_loops = down_expr->get_loop_ids();
    // If upper expression is out of Loop, we can insert Buffer implicitly after him
    if (up_loops.empty()) {
        const auto it = std::find(linear_ir.cbegin(), linear_ir.cend(), up_expr);
        OPENVINO_ASSERT(it != linear_ir.cend(), "Upper expression hasn't been found to insert Buffer after him!");
        return std::next(it);
    }
    // If lower expression is out of Loop, we can insert Buffer implicitly before him
    if (down_loops.empty()) {
        const auto it = std::find(linear_ir.cbegin(), linear_ir.cend(), down_expr);
        OPENVINO_ASSERT(it != linear_ir.cend(), "Lower expression hasn't been found to insert Buffer after him!");
        return it;
    }

    const auto up_loop_count = up_loops.size();
    const auto down_loop_count = down_loops.size();
    size_t loop_idx = 0;
    for (; loop_idx < std::min(up_loop_count, down_loop_count); ++loop_idx) {
        if (up_loops[loop_idx] != down_loops[loop_idx])
            break;
    }
    // If upper expression is inside Loop, we should insert Buffer after this Loop
    if (loop_idx < up_loop_count) {
        const auto up_loop_id = up_loops[loop_idx];
        const auto loop_info = loop_manager->get_loop_info(up_loop_id);
        LinearIR::constExprIt loop_begin_pos, loop_end_pos;
        loop_manager->get_loop_bounds(linear_ir, up_loop_id, loop_begin_pos, loop_end_pos);
        return loop_end_pos;
    }
    // If lower expression is inside Loop, we should insert Buffer before this Loop
    if (loop_idx < down_loop_count) {
        const auto down_loop_id = down_loops[loop_idx];
        const auto loop_info = loop_manager->get_loop_info(down_loop_id);
        LinearIR::constExprIt loop_begin_pos, loop_end_pos;
        loop_manager->get_loop_bounds(linear_ir, down_loop_id, loop_begin_pos, loop_end_pos);
        return loop_begin_pos;
    }
    OPENVINO_THROW("Incorrect configuration for Buffer insertion!");
}

void InsertBuffers::insertion(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager,
                              const std::vector<LinearIR::LoopManager::LoopPort>& loop_entries,
                              const std::vector<LinearIR::LoopManager::LoopPort>& loop_exits) {
    for (const auto& entry_point : loop_entries) {
        const auto& entry_port = entry_point.expr_port;
        const auto& expr = entry_port->get_expr();
        const auto port = entry_port->get_index();
        const auto node = expr->get_node();
        const auto& input_connector = expr->get_input_port_connector(port);
        const auto& parent_expr_output = input_connector->get_source();
        const auto& parent_expr = parent_expr_output.get_expr();
        const auto parent_port = parent_expr_output.get_index();
        const auto parent = parent_expr->get_node();
        if (ov::is_type<op::Buffer>(parent) ||
            ov::is_type<op::VectorBuffer>(parent) ||
            ov::is_type<ov::op::v0::Parameter>(parent) ||
            ov::is_type<ov::op::v0::Constant>(parent))
            continue;

        // Each MemoryAccess op needs Buffer
        const auto parent_ma = ov::as_type_ptr<op::MemoryAccess>(parent);
        const auto node_ma = ov::as_type_ptr<op::MemoryAccess>(node);
        bool is_buffer_needed = (parent_ma && parent_ma->is_memory_access_output_port(parent_port)) ||
                                (node_ma && node_ma->is_memory_access_input_port(port));
        if (!is_buffer_needed) {
            const auto current_loops = expr->get_loop_ids();
            const auto parent_loops = parent_expr->get_loop_ids();
            const auto current_loop_count = current_loops.size();
            const auto parent_loop_count = parent_loops.size();
            for (size_t i = 0; i < std::min(current_loop_count, parent_loop_count); ++i) {
                if (current_loops[i] != parent_loops[i]) {
                    is_buffer_needed = true;
                    break;
                }
            }
        }

        if (is_buffer_needed) {
            // We should insert Buffer between first different Loops.
            // Example: Target Parent Loop identifies: 3, 2, 1
            //          Current expr Loop identifies:  3, 4, 6
            //          Need to insert between 2nd and 4th Loops - after 2nd Loop
            const auto pos = insertion_position(linear_ir, loop_manager, parent_expr, expr);
            const auto buffer = std::make_shared<op::Buffer>(parent->output(parent_port), m_buffer_allocation_rank);
            PortDescriptorUtils::set_port_descriptor_ptr(buffer->output(0), parent_expr_output.get_descriptor_ptr()->clone());
            // Output connector is automatically filled from PortDescriptor
            const auto buffer_expr = linear_ir.create_expression(buffer, {input_connector});
            linear_ir.insert(pos, buffer_expr);
            linear_ir.replace_input(*entry_port.get(), buffer_expr->get_output_port_connector(0));
        }
    }

    for (const auto& exit_point : loop_exits) {
        const auto& exit_port = exit_point.expr_port;
        const auto& expr = exit_port->get_expr();
        const auto port = exit_port->get_index();
        const auto node = expr->get_node();
        const auto output_connector = exit_port->get_port_connector_ptr();
        const auto child_exprs_inputs = output_connector->get_consumers();
        const auto current_loops = expr->get_loop_ids();
        const auto current_loop_count = current_loops.size();
        const std::vector<PortConnectorPtr> node_outs = {output_connector};

        std::set<ExpressionPort> potential_consumers;
        std::set<ExpressionPtr> buffers;
        for (const auto& child_expr_input : child_exprs_inputs) {
            const auto& child_expr = child_expr_input.get_expr();
            const auto child_port = child_expr_input.get_index();
            const auto& child = child_expr->get_node();
            if (ov::is_type<ov::op::v0::Result>(child))
                continue;
            if (ov::is_type<op::Buffer>(child)) {
                buffers.insert(child_expr);
                continue;
            }
            // Each MemoryAccess op needs Buffer
            const auto child_ma = ov::as_type_ptr<op::MemoryAccess>(child);
            const auto node_ma = ov::as_type_ptr<op::MemoryAccess>(node);
            if ((child_ma && child_ma->is_memory_access_input_port(child_port)) ||
                (node_ma && node_ma->is_memory_access_output_port(port))) {
                potential_consumers.insert(child_expr_input);
                continue;
            }

            const auto child_loops = child_expr->get_loop_ids();
            const auto child_loop_count = child_loops.size();
            for (size_t i = 0; i < std::min(current_loop_count, child_loop_count); ++i) {
                if (current_loops[i] != child_loops[i]) {
                    potential_consumers.insert(child_expr_input);
                    break;
                }
            }
        }

        if (!potential_consumers.empty() || buffers.size() > 1) {
            // If some of children from one common port are different Buffers,
            // we should remove them to insert one common Buffer on one common port
            if (!buffers.empty()) {
                for (const auto& buffer : buffers) {
                    const auto& buffer_out = buffer->get_output_port_connector(0);
                    const auto buffer_consumers_inputs = buffer_out->get_consumers();
                    linear_ir.replace_input(buffer_consumers_inputs, output_connector);
                    potential_consumers.insert(buffer_consumers_inputs.begin(), buffer_consumers_inputs.end());
                    linear_ir.erase(std::find(linear_ir.begin(), linear_ir.end(), buffer));
                }
            }

            // We should insert Buffer between first different Loops.
            // Example: Current expr Loop identifies: 3, 2, 1
            //          Target consumers Loop identifies:  3, 4, 6
            //          Need to insert after 2nd Loops
            // Note: All potential consumers must have the same count of first equal Loop identifies and the same count of different last identifies
            // TODO: Need to verify that
            const auto pos = insertion_position(linear_ir, loop_manager, expr, (*potential_consumers.begin()).get_expr());

            auto buffer = std::make_shared<op::Buffer>(node->output(port), m_buffer_allocation_rank);
            PortDescriptorUtils::set_port_descriptor_ptr(buffer->output(0), exit_port->get_descriptor_ptr()->clone());
            // We cannot insert Node output connector on Buffer output because not all consumers of Node needs Buffer
            //  Example:
            //       Add
            //      /   \  <- It should be the same PortConnector
            //  Result   Buffer
            //             |    <- It should be new PortConnector
            //            Relu
            // Output port connector is automatically filled from PortDescriptor
            const auto buffer_expr = linear_ir.create_expression(buffer, node_outs);
            linear_ir.insert(pos, buffer_expr);
            linear_ir.replace_input(potential_consumers, buffer_expr->get_output_port_connector(0));
        }
    }
}

bool InsertBuffers::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertBuffers")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto loop_data_map = loop_manager->get_map();
    for (const auto& loop_data : loop_data_map) {
        const auto loop_info = loop_data.second;
        const auto loop_entries = loop_info->entry_points;
        const auto loop_exits = loop_info->exit_points;
        insertion(linear_ir, loop_manager, loop_entries, loop_exits);
    }

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto node = (*expr_it)->get_node();
        const auto ma = ov::as_type_ptr<op::MemoryAccess>(node);
        if (!ma)
            continue;

        const auto input_ports = ma->get_memory_access_input_ports();
        const auto output_ports = ma->get_memory_access_output_ports();
        std::vector<LinearIR::LoopManager::LoopPort> loop_entries(input_ports.size()), loop_exits(output_ports.size());
        for (const auto& p : input_ports) {
            loop_entries[p.first] = expr->get_input_port(p.first);
        }
        for (const auto& p : output_ports) {
            loop_exits[p.first] = expr->get_output_port(p.first);
        }

        insertion(linear_ir, loop_manager, loop_entries, loop_exits);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
