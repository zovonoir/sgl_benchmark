cd /home/jialzhu/deepseekv4/sgl_benchmark
#python3 -m vllm_bench --config /home/jialzhu/deepseekv4/sgl_benchmark/vllm_bench/config_examples/config_deepseek_v4_exp.yaml --run-mode benchmark
#sleep 10
#docker restart zov-dsv4
#
#python3 -m vllm_bench --config /home/jialzhu/deepseekv4/sgl_benchmark/vllm_bench/config_examples/config_deepseek_v4_disable_custom_allreduce.yaml  --run-mode benchmark
#sleep 10
#docker restart zov-dsv4
#
#
#python3 -m vllm_bench --config /home/jialzhu/deepseekv4/sgl_benchmark/vllm_bench/config_examples/config_deepseek_quick_allreduce_fp.yaml   --run-mode benchmark
#sleep 10
#docker restart zov-dsv4
#
#
#python3 -m vllm_bench --config /home/jialzhu/deepseekv4/sgl_benchmark/vllm_bench/config_examples/config_deepseek_quick_all_reduce_int4.yaml   --run-mode benchmark
#sleep 10
#docker restart zov-dsv4
#
#
#python3 -m vllm_bench --config /home/jialzhu/deepseekv4/sgl_benchmark/vllm_bench/config_examples/config_deepseek_quick_all_reduce_int6.yaml   --run-mode benchmark
#sleep 10
#docker restart zov-dsv4
#
#
#python3 -m vllm_bench --config /home/jialzhu/deepseekv4/sgl_benchmark/vllm_bench/config_examples/config_deepseek_quick_all_reduce_int8.yaml   --run-mode benchmark
#sleep 10
#docker restart zov-dsv4

#python3 -m vllm_bench --config /home/jialzhu/deepseekv4/sgl_benchmark/vllm_bench/config_examples/config_deepseek_tp8ep8_baseline.yaml --run-mode benchmark
#sleep 10
#docker restart zov-dsv4

python3 -m vllm_bench --config /home/jialzhu/deepseekv4/sgl_benchmark/vllm_bench/config_examples/config_deepseek_tp8ep8_mori_high_throughput.yaml --run-mode benchmark
sleep 10
docker restart zov-dsv4
