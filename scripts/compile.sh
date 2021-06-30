make -C build -j faiss_avx2 swigfaiss_avx2
cd build/faiss/python
python setup.py build
cd ../../../

