export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
#export LD_LIBRARY_PATH=/workspace/rccl/build/
# export LD_LIBRARY_PATH="/workspace/mscclpp/build:/workspace/mscclpp/build/apps/nccl:${LD_LIBRARY_PATH}"

torchrun --standalone --nnodes=1 --nproc-per-node=2 test_broadcast.py 
