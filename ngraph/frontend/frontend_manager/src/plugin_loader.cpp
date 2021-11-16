// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <Windows.h>
#    include <direct.h>
#else  // _WIN32
#    include <dirent.h>
#    include <dlfcn.h>
#    include <unistd.h>
#endif  // _WIN32

#include <sys/stat.h>

#include <ngraph/log.hpp>
#include <string>
#include <vector>

#include "openvino/util/file_util.hpp"
#include "plugin_loader.hpp"

using namespace ov;
using namespace ov::frontend;

#ifdef WIN32
#    define DLOPEN(file_str) LoadLibrary(TEXT(file_str.c_str()))
#    define DLSYM(obj, func) GetProcAddress(obj, func)
#    define DLCLOSE(obj)     FreeLibrary(obj)
#    define DLERROR()        ""
#else
#    define DLOPEN(file_str) dlopen(file_str.c_str(), RTLD_LAZY)
#    define DLSYM(obj, func) dlsym(obj, func)
#    define DLCLOSE(obj)     dlclose(obj)
#    define DLERROR()        dlerror()
#endif

// TODO: change to std::filesystem for C++17
static std::vector<std::string> list_files(const std::string& path) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    std::vector<std::string> res;
    try {
        ov::util::iterate_files(
            path,
            [&res](const std::string& file, bool is_dir) {
                if (!is_dir && file.find("_ngraph_frontend") != std::string::npos) {
#ifdef _WIN32
                    std::string ext = ".dll";
#elif defined(__APPLE__)
                    std::string ext = ".dylib";
#else
                    std::string ext = ".so";
#endif
                    if (file.find(ext) != std::string::npos) {
                        res.push_back(file);
                    }
                }
            },
            // ilavreno: this is current solution for static runtime
            // since frontends are still dynamic libraries and they are located in
            // a different folder with compare to frontend_manager one (in ./lib)
            // we are trying to use recursive search. Can be reverted back in future
            // once the frontends are static too.
            true,
            true);
    } catch (...) {
        // Ignore exceptions
    }
    return res;
    NGRAPH_SUPPRESS_DEPRECATED_END
}

std::vector<PluginData> ov::frontend::load_plugins(const std::string& dir_name) {
    auto files = list_files(dir_name);
    std::vector<PluginData> res;
    for (const auto& file : files) {
        auto shared_object = DLOPEN(file);
        if (!shared_object) {
            NGRAPH_DEBUG << "Error loading FrontEnd " << file << " " << DLERROR() << std::endl;
            continue;
        }

        PluginHandle guard([shared_object, file]() {
            DLCLOSE(shared_object);
        });

        auto info_addr = reinterpret_cast<void* (*)()>(DLSYM(shared_object, "GetAPIVersion"));
        if (!info_addr) {
            continue;
        }
        FrontEndVersion plug_info{reinterpret_cast<FrontEndVersion>(info_addr())};

        if (plug_info != OV_FRONTEND_API_VERSION) {
            // Plugin has incompatible API version, do not load it
            continue;
        }

        auto creator_addr = reinterpret_cast<void* (*)()>(DLSYM(shared_object, "GetFrontEndData"));
        if (!creator_addr) {
            continue;
        }

        std::unique_ptr<FrontEndPluginInfo> fact{reinterpret_cast<FrontEndPluginInfo*>(creator_addr())};

        res.push_back(PluginData(std::move(guard), std::move(*fact)));
    }
    return res;
}
