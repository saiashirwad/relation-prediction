mkdir release
g++ ./base/Base.cpp -fPIC -shared -o ./release/Base_Linux.so -pthread -O3 -march=native
