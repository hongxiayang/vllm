docker build --build-arg BASE_IMAGE="compute-artifactory.amd.com:5000/rocm-plus-docker/framework/release-public:rocm6.0_ubuntu20.04_py3.9_pytorch_rocm6.0_internal_testing" \
--build-arg ARG_GFX_ARCHS="gfx90a;gfx1100;gfx942" -f Dockerfile.test -t test_args .

# --build-arg LLVM_GFX_ARCHS="gfx90a;gfx1100;gfx942" -f Dockerfile.test -t test_args .


#docker build -f Dockerfile.test -t test_args .

