
log_file=msb-scan.log
rm "$log_file"

scan() {
    export OV_MSB_SCAN=$1
    # result_file="msb-scan-result-$1.log"
    # rm "$result_file"
    printf "=============\n\n$1\n\n=============\n" | tee -a "$log_file"
    ./bin/intel64/Release/ov_transformations_tests --gtest_filter=*scan_cv_bench_cache | tee -a "$log_file" | \
        sed -n 's/MSB // p' | tee "msb-scan-result-$1.log"
    unset OV_MSB_SCAN
    echo | tee -a "$log_file"
}

scan MATMUL_ADD
scan IDENTICAL
scan MATMUL
