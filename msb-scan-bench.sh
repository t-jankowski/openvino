# model_path=/home/tj/space/bert-base-ner-fp16/bert-base-ner.xml

nireq=10
niter=1000

log_file=msb-scan-bench-full.log
result_log=msb-scan-bench-result.csv
rm "$log_file"

run-single() {
    echo >> "$log_file"
    comand="/home/tj/ov/bin/intel64/Release-bench/benchmark_app -m $1 -d $2 -nireq $nireq -niter $niter"
    tee_log="tee -a $log_file"
    sed_fps='s/\[ INFO \] Throughput: +(.+) FPS/\1/ p'

    export OV_MERGE_MATMULS_BENCHMARK=1
    v1=$( $comand | $tee_log | sed -En "$sed_fps" )

    echo >> "$log_file"

    unset OV_MERGE_MATMULS_BENCHMARK
    v2=$( $comand | $tee_log | sed -En "$sed_fps" )

    printf "$1,$v1,$v2\n" | tee -a $result_log

    echo >> "$log_file"
    echo '---------' >> "$log_file"
}

run-bunch() {
    for f in $(cat $2); do
        run-single $f $1
    done
}

echo "device CPU" | tee $result_log
echo nireq $nireq, niter $niter | tee $result_log

echo 'model,w/,w/o' | tee $result_log
run-bunch CPU /home/tj/space/bert-base-ner-fp16/msb-scan-result-IDENTICAL.log


# sed -i '$ d' "$log_file"
# sed -i '$ d' "$log_file"
