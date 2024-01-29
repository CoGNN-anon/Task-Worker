#include "troy/app/TroyFHEWrapper.cuh"
#include <string>
#include <vector>
#include <iostream>

using namespace troyn;
using namespace std;

const uint64_t MultDataBound = (1ul << 44);
const uint64_t MultMod = (1ul << 44);
const uint64_t MultModBithLength = 44;

template <typename T>
void print_vector(const std::vector<T>& v, size_t num = 0) {
  size_t cnt = 0;
  for (auto elem : v) {
    if (num != 0 && cnt >= num) break;
    std::cout << elem << " ";
    cnt++;
  }
  std::cout << "\n";
}

std::vector<uint64_t> add_vector(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    vector<uint64_t> c(a.size(), 0);
    for (int i = 0; i < c.size(); ++i) {
        c[i] = (a[i] + b[i]) % MultMod;
    }
    return c;
}

void test_mul_scaler_sub_random(FHEWrapper& fhe) {
    size_t dim = 1433;
    // std::vector<uint64_t> input = {1, 2, 3, 4, 5};
    std::vector<uint64_t> input(dim, 10);
    uint64_t scaler = 10;
    for (int i = 0; i < 3000; ++i) {
        auto input_cipher = fhe.encrypt(input);
        std::vector<uint64_t> random = fhe.mul_scaler_and_subtract_random(input_cipher, scaler, dim);
        std::vector<uint64_t> output = fhe.decrypt(input_cipher, dim);        
    }
    // printf("output size %lu\n", output.size());
    // print_vector(output);
    // printf("random size %lu\n", random.size());
    // print_vector(random);
    // printf("added size %lu\n", add_vector(output, random).size());
    // print_vector(add_vector(output, random));
}

int main() {
    auto fhe = FHEWrapper();
    test_mul_scaler_sub_random(fhe);
    return 0;
}