#include <tuple>
#include <vector>
#include <string>
#include <utility>

#include "utils.cuh"


std::tuple<std::vector<std::string>, std::vector<std::vector<std::string>>> loop_bench_hdim128_c1();
std::tuple<std::vector<std::string>, std::vector<std::vector<std::string>>> loop_bench_hdim128_c0();
std::tuple<std::vector<std::string>, std::vector<std::vector<std::string>>> loop_bench_hdim64_c1();
std::tuple<std::vector<std::string>, std::vector<std::vector<std::string>>> loop_bench_hdim64_c0();


int main(int argc, char* argv[]){
  std::string csv_filename = parse_filename_arg(argc, argv);

  auto [hdim64_c0_csv_header, hdim64_c0_csv_data] = loop_bench_hdim64_c0();
  auto [hdim128_c0_csv_header, hdim128_c0_csv_data] = loop_bench_hdim128_c0();
  auto [hdim64_c1_csv_header, hdim64_c1_csv_data] = loop_bench_hdim64_c1();
  auto [hdim128_c1_csv_header, hdim128_c1_csv_data] = loop_bench_hdim128_c1();


  std::vector<std::vector<std::string>> csv_data;
  for (auto & row : hdim64_c0_csv_data) {
    csv_data.push_back(row);
  }
  for (auto & row : hdim128_c0_csv_data) {
    csv_data.push_back(row);
  }
  for (auto & row : hdim64_c1_csv_data) {
    csv_data.push_back(row);
  }
  for (auto & row : hdim128_c1_csv_data) {
    csv_data.push_back(row);
  }

  if (!csv_filename.empty()) {
    write_result_to_csv(csv_filename, hdim128_c1_csv_header, csv_data);
  }
}