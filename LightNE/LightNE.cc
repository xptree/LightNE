// Usage:
// numactl -i all ./GraphEmbed -walksperedge 100 -walklen 10 -s -m -rounds 3 twitter_SJ
// flags:
//   required: //   optional:
//     -walksperedge: the number of random walks to perform per vertex
//     -walklen: the length of the random walk
//     -rounds : the number of times to run the algorithm //     -c : indicate that the graph is compressed
//     -m : indicate that the graph should be mmap'd
//     -s : indicate that the graph is symmetric

#include <cassert>
#include <cstdio>
#include <typeinfo>
#include <vector>
#include "PathEmbed.h"
#include "SpectralPropagation.h"

template <typename FP>
void save(const std::string& file, FP* emb, size_t n, size_t d) {
  timer t_save; t_save.start();
	// write buffer to file
	if(std::FILE* f = std::fopen(file.c_str(), "wb")) {
			std::fwrite(emb, sizeof(FP), n*d, f);
			std::fclose(f);
	}
  t_save.stop(); t_save.reportTotal("save time");
}

template <class Graph>
double LightNE_mkl_runner(Graph& GA, commandLine P) {
  size_t walks_per_edge = static_cast<size_t>(P.getOptionLongValue("-walksperedge", 100));
  size_t walk_len = static_cast<size_t>(P.getOptionLongValue("-walklen", 10));
  size_t table_size = static_cast<size_t>(P.getOptionLongValue("-tablesz", 0));
  size_t rank = static_cast<size_t>(P.getOptionLongValue("-rank", 256));
  size_t dim = static_cast<size_t>(P.getOptionLongValue("-dim", 128));
  size_t order = static_cast<size_t>(P.getOptionLongValue("-order", 10));
  size_t negative = static_cast<size_t>(P.getOptionLongValue("-negative", 1));
  bool upper = static_cast<bool>(P.getOptionLongValue("-upper", 0));
  bool sample = static_cast<bool>(P.getOptionLongValue("-sample", 0));
  bool analyze = static_cast<bool>(P.getOptionLongValue("-analyze", 0));
  bool normalize = static_cast<bool>(P.getOptionLongValue("-normalize", 0));
  float sample_ratio = static_cast<float>(P.getOptionDoubleValue("-sample_ratio", 1.0));
  float mem_ratio = static_cast<float>(P.getOptionDoubleValue("-mem_ratio", 1.0));
  std::string ne_method = P.getOptionValue("-ne_method", "netsmf");
  std::string ne_out = P.getOptionValue("-ne_out", "");
  std::string pro_out = P.getOptionValue("-pro_out", "");
  std::string step_coeff_str = P.getOptionValue("-step_coeff", "1,1,1,1,1,1,1,1,1,1");
  bool random_project_only = static_cast<bool>(P.getOptionLongValue("-random_project_only", 0));
  bool sparse_project = static_cast<bool>(P.getOptionLongValue("-sparse_project", 0));
  float sparse_project_s = static_cast<float>(P.getOptionDoubleValue("-sparse_project_s", 100.0));

  std::vector<float> step_coeff;
  std::stringstream ss(step_coeff_str);
  while (ss.good()) {
    std::string substr;
    std::getline(ss, substr, ',');
    float coeff = std::stof(substr);
    step_coeff.push_back(coeff);
  }

  assert(ne_method == "netsmf" || ne_method == "ne_zhang_et_al");
  assert(step_coeff.size() == walk_len);

  std::cout << "### Application: LightNE_mkl" << std::endl;
  std::cout << "### Graph: " << P.getArgument(0) << std::endl;
  std::cout << "### Threads: " << num_workers() << std::endl;
  std::cout << "### n: " << GA.n << std::endl;
  std::cout << "### m: " << GA.m << std::endl;
  std::cout << "### Params: " << std::endl;
  std::cout << "###  -walksperedge = " << walks_per_edge       << std::endl;
  std::cout << "###  -walklen = " << walk_len       << std::endl;
  std::cout << "###  -tablesz = " << table_size       << std::endl;
  std::cout << "###  -rank = " << rank       << std::endl;
  std::cout << "###  -dim = " << dim       << std::endl;
  std::cout << "###  -order = " << order       << std::endl;
  std::cout << "###  -negative = " << negative       << std::endl;
  std::cout << "###  -upper = " << std::boolalpha << upper       << std::endl;
  std::cout << "###  -sample = " << std::boolalpha << sample       << std::endl;
  std::cout << "###  -analyze = " << std::boolalpha << analyze << std::endl;
  std::cout << "###  -ne_method = " << ne_method << " (nestmf: see Qiu etal: arxiv.org/abs/1906.11156, ne_zhang_et_al: see sec. 3.1 of Zhang et al: www.ijcai.org/proceedings/2019/594)"       << std::endl;
  std::cout << "###  -normalize = " << normalize << std::endl;
  std::cout << "###  -sample_ratio = " << sample_ratio << std::endl;
  std::cout << "###  -mem_ratio = " << mem_ratio << std::endl;
  std::cout << "###  -random_project_only = " << random_project_only << std::endl;
  std::cout << "###  -sparse_project = " << sparse_project << std::endl;
  std::cout << "###  -sparse_project_s = " << sparse_project_s << std::endl;
  std::cout << "###  -ne_out = " << ne_out       << std::endl;
  std::cout << "###  -pro_out = " << pro_out       << std::endl;
  std::cout << "###  -step_coeff = ";
  for (float coeff: step_coeff) {
    std::cout << coeff << " ";
  }
  std::cout << std::endl;
  std::cout << "### ------------------------------------" << std::endl;

  timer t; t.start();

  // path_embed::setup_link_prediction(GA, 0.0000001, "cw_edge.txt", "cw_degree.txt");

  using FP = float;
  std::cout << "# using float point type " << typeid(FP).name() << " (f for float, d for double)" << std::endl;
  FP* emb = NULL;
  if (ne_method == "netsmf") {
    emb = path_embed::NetSMF<Graph, FP>(GA, walks_per_edge, walk_len, upper, sample, rank, dim, analyze, table_size, negative, normalize, sample_ratio, mem_ratio, step_coeff, random_project_only, sparse_project, sparse_project_s);
  } else {
    emb = path_embed::NE_Zhang_et_al<Graph, FP>(GA, rank, dim);
  }
  if (ne_out.size() > 0) {
    std::cout << "dump network embedding to " << ne_out << std::endl;
    save<FP>(ne_out, emb, GA.n, dim);
  }
  if (pro_out.size() > 0) {
    timer t_pro; t_pro.start();
    FP* pronemb = spectral_propagation::chebyshev_expansion<Graph, FP>(emb, GA, dim, dim, order, 0.2, 0.5);
    t_pro.stop(); t_pro.reportTotal("proX time");
    std::cout << "dump proX embedding to " << pro_out << std::endl;
    save<FP>(pro_out, pronemb, GA.n, dim);
  }
  double tt =  t.stop();
  std::cout << "### Running Time: " << tt << std::endl;
  return tt;
}

generate_symmetric_main(LightNE_mkl_runner, false);
