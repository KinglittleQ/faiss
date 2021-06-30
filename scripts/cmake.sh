cmake -B build \
    -DFAISS_ENABLE_GPU=OFF -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_OPT_LEVEL=avx2 \
    -DMKL_LIBRARIES=/opt/conda/lib \
    -DPython_EXECUTABLE=$(which python) \
    .
