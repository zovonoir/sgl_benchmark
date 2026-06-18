# vllm
docker run -it --name zov-dsv4-0615-2 \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v /mnt:/.cache/huggingface \
	-v /home/jialzhu:/dockerx \
	--cap-add=SYS_PTRACE --privileged  \
	--device /dev/dri:/dev/dri --device /dev/kfd:/dev/kfd \
    --ipc=host \
	--network=host \
    --shm-size=128g \
    --tmpfs /model_ram:size=1024G,mode=1777 \
	--entrypoint /bin/bash \
    -t sabreshao/vllm:dsv4_0615n



# container_env:
#   - "HF_HOME=/.cache/huggingface/"
#   - "HF_HUB_CACHE=/.cache/huggingface"
#   - "VLLM_ROCM_USE_AITER=1"
#   - "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
#   - "VLLM_ROCM_USE_AITER_MOE=1"
#   - "SNAP=/.cache/huggingface/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/89d501aed998d33fa4f4702102ec1bb2331e10f6"
#   - "HF_HUB_OFFLINE=1"
#   - "TRANSFORMERS_OFFLINE=1"