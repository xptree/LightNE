#pragma once

#include "ligra.h"
#include "pbbslib/random.h"
#include "pbbslib/sequence_ops.h"
#include "pbbslib/monoid.h"
#include "pbbslib/binary_search.h"
#include "pbbslib/sample_sort.h"
#include "par_table.h"
#include "maybe.h"

#include "ligra/edge_map_reduce.h"
#include "ligra/graph_mutation.h"

#include <math.h>
#include "MKLSVD.h"
#define EPS 1e-5

#include <iostream>
#include <fstream>
#include <random>
#include <vector>

#define TWO63 0x8000000000000000u
#define TWO64f (TWO63*2.0)
double map_uint64_t(uint64_t u) {
  double y = (double) u;
  return y/TWO64f;
}


namespace path_embed {
  using K = size_t;
  using V = double; // 16 byte aligned KVs

  template <class Graph, class Rand>
  inline uintE random_walk(Graph& GA,
      uintE u, size_t walk_len, Rand& seed) {
    for (size_t i=0; i<walk_len; i++) {
      auto vtx = GA.get_vertex(u);
      uintE out_degree = vtx.getOutDegree();
      size_t ngh_id = seed.rand() % out_degree; // mod is costly
      auto [ngh, wgh] = vtx.get_ith_out_neighbor(u, ngh_id); // update node
      u = ngh;
      seed = seed.next();
    }
    return u;
  }

  template <class Graph, class Rand>
  std::tuple<uintE, uintE> path_sample(Graph& GA, uintE u, uintE v,
      Rand& seed, size_t walk_len = 10) {
    uintE k = seed.rand() % walk_len + 1; seed = seed.next();
    u = random_walk(GA, u, k-1, seed);
    v = random_walk(GA, v, walk_len-k, seed);
    return std::make_tuple(u,v);
  }

  /* Should not use---uses a ton of memory and prohibits running on large
   * compressed graphs */
  template <class Graph>
  auto get_graph_edges(Graph& GA) {
    using W = typename Graph::weight_type;
    auto offs = pbbs::sequence<size_t>(GA.n, [&] (size_t i) { return GA.get_vertex(i).getOutDegree(); });
    size_t m = pbbslib::scan_add_inplace(offs.slice());
    assert(GA.m == m);

    auto edges = pbbs::sequence<std::tuple<uintE, uintE>>(m);
    parallel_for(0, GA.n, [&] (size_t i) {
      size_t k = 0;
      size_t off_i = offs[i];
      auto map_f = [&] (const uintE& u, const uintE& v, const W& wgh) {
        edges[off_i + k++] = std::make_tuple(u, v);
      };
      GA.get_vertex(i).mapOutNgh(i, map_f, false);
    });
    return std::move(edges);
  }

  template <class Graph>
  auto generate_trunc_log_matrix_v2(Graph& GA,
      pbbs::random seed, size_t walks_per_edge, size_t walk_len, bool upper, bool sample, size_t table_size, size_t negative, float sample_ratio, float mem_ratio, const std::vector<float>& step_coeff) {
    using W = typename Graph::weight_type;
    size_t n = GA.n;
    size_t m = GA.m;
    double logn = log(n);
    std::cout << "# logn = " << logn << " sample_ratio = " << sample_ratio << " mem_ratio = " << mem_ratio << std::endl;

    using T = std::tuple<K, V>;
    auto empty = std::make_tuple(std::numeric_limits<size_t>::max(), static_cast<V>(0));
    if (sample) {
      table_size = std::max(static_cast<size_t>(walk_len * GA.n * logn * mem_ratio), table_size);
    } else {
      table_size = std::max(static_cast<size_t>(walk_len * GA.m * mem_ratio), table_size);
    }
    std::cout << "# table_size = " << table_size << std::endl;
    auto hash_f = [&] (size_t k) { return pbbs::hash64_2(k); };
    auto HT = make_par_table(table_size, empty, hash_f, 2);
    std::cout << "# actual table.size = " << HT.m << std::endl;
    std::cout << "# empty = (" << std::get<0>(HT.empty) << "," << std::get<1>(HT.empty) << ")" << std::endl;
    std::cout << "# empty key = " << HT.empty_key << std::endl;

    auto M = (sample) ? (walk_len * GA.n * logn * sample_ratio) : (walk_len * walks_per_edge * GA.m * sample_ratio);
    double one_over_lambda = static_cast<double>(M) / static_cast<double>(m);
    size_t exp_rv_rounded = static_cast<size_t>(one_over_lambda); // this is one option---don't sample, and just take expectation
    double fractional_bits = one_over_lambda - static_cast<double>(exp_rv_rounded);

    std::cout << "# total samples = " << M << std::endl;
    std::cout << "# sample param = " << sample << std::endl;
    std::cout << "# 1/lambda = " << one_over_lambda << std::endl;
    std::cout << "# exp_rv_rounded = " << exp_rv_rounded << std::endl;
    std::cout << "# fractional_bits = " << fractional_bits << std::endl;

    /* Note: This is the version where we generate ALL of the samples at
     * the same time in parallel. Measuring #failed cases indicates that
     * surprisingly, we are not incurring a ton of contention in the table. */
    auto emit_ht = [&] (const uintE& x, const uintE& z, const V& w) {
      size_t key = (static_cast<size_t>(std::min(x,z)) << 32UL) +
                    static_cast<size_t>(std::max(x,z));
      HT.insert_add(std::make_tuple(key, w));
    };
    timer t; t.start();
    auto g_map_f = [&] (const uintE& u, const uintE& v, const W& wgh) {
      //1. generate exp r.v. seeded by (u,v)
      size_t uv = (static_cast<size_t>(u) << 32UL) | static_cast<size_t>(v);
      auto our_seed = seed.fork(uv);
      // a simpler version than sampling from a Poisson distribution
      size_t exp_rv = exp_rv_rounded;
      if (fractional_bits > 0.0000000001) { /* somewhat non-trivial? */
        exp_rv += static_cast<double>(map_uint64_t(our_seed.rand()) < fractional_bits);
        our_seed = our_seed.next();
      }

      if (!sample) {
        // Apply the sampler to this edge exp_rv many times
        for (size_t k=0; k<exp_rv; k++) {
          // uintE r = distribution(generator) + 1;
          uintE r = our_seed.rand() % walk_len + 1;
          our_seed = our_seed.next();
          auto [x, z] = path_sample(GA, u, v, our_seed, r);
          emit_ht(x, z, static_cast<V>(1));
        }
      } else {
        double approx_effective_resistance = static_cast<double>(1.0
            /  GA.get_vertex(u).getOutDegree()
            + 1.0 / GA.get_vertex(v).getOutDegree())
            * logn;
        approx_effective_resistance = std::min(approx_effective_resistance, 1.0);
        double ipw = 1.0 / approx_effective_resistance;
        // Apply the sampler to this edge exp_rv many times
        for (size_t k=0; k<exp_rv; k++) {
          // apply eff-resist sampling
          double eff_prob = map_uint64_t(our_seed.rand());
          our_seed = our_seed.next();
          if (eff_prob < approx_effective_resistance) {
            // uintE r = distribution(generator) + 1;
            uintE r = our_seed.rand() % walk_len + 1;
            our_seed = our_seed.next();
            auto [x, z] = path_sample(GA, u, v, our_seed, r);
            emit_ht(x, z, static_cast<V>(ipw));
          }
        }
      }
    };
    GA.map_edges(g_map_f);
    t.stop(); t.reportTotal("# count time");

    auto wgh_seq = pbbslib::make_sequence<V>(HT.m, [&] (size_t i) {
      if (std::get<0>(HT.table[i]) != std::numeric_limits<size_t>::max()) {
        return std::get<1>(HT.table[i]);
      }
      return static_cast<V>(0);
    });
    auto distinct_seq = pbbslib::make_sequence<V>(HT.m, [&] (size_t i) {
      if (std::get<0>(HT.table[i]) != std::numeric_limits<size_t>::max()) {
        return static_cast<V>(1);
      }
      return static_cast<V>(0);
    });
    V M_exact = pbbs::reduce(wgh_seq, pbbs::addm<V>());
    std::cout << "# tot wgh (samples) = " << M_exact << std::endl;
//    // following are for debugging
//    double M_distinct = pbbs::reduce(distinct_seq, pbbs::addm<size_t>());
//    size_t avg_weight = static_cast<double>(M_exact) / static_cast<double>(M_distinct);
//    std::cout << "# distinct samples = " << M_distinct << std::endl;
//    std::cout << "# max wgh = " << pbbs::reduce(wgh_seq, pbbs::maxm<size_t>()) << std::endl;
//    std::cout << "# avg_weight = " << avg_weight << std::endl;

    // (2) Prune based on trunc_log.
    // auto my_log = [&](double x) -> double {return log(x);};

    V vol = GA.m; // \sum_{v \in V} d_v
    V factor = vol * m / M_exact / 2 / negative;

    std::cout << "# factor = " << factor << std::endl;
    auto map_f = [&] (T kv) {
      uintE u = (uintE)(std::get<0>(kv) >> 32UL);
      uintE v = (uintE)(std::get<0>(kv));
      // if (u == v) { return empty; }
      uintE d_u = GA.get_vertex(u).getOutDegree();
      uintE d_v = GA.get_vertex(v).getOutDegree();
      V wgh = (u!=v) ? std::get<1>(kv) : std::get<1>(kv)*2;
      V adjusted_wgh = wgh * factor / d_u / d_v;
      V log_wgh = log(adjusted_wgh);
      if (log_wgh > 0.0) {
        return std::make_tuple(std::get<0>(kv), log_wgh);
      } else {
        return empty;
      }
    };
    HT.map_table(map_f);

    std::cout << "# create sparsifier from hash table" << std::endl;
    auto sparsifier_upper_entries = HT.entries();
    HT.del();
    std::cout << "# tot kept = " << sparsifier_upper_entries.size() << " in upper triangle" << std::endl;

    auto t_less = [&] (const T& l, const T& r) {
      return std::get<0>(l) < std::get<0>(r);
    };

    if (upper) {
      std::cout << "# return upper triangle only" << std::endl;
      timer st; st.start();
      std::cout << "# starting sort" << std::endl;
      pbbs::sample_sort_inplace(sparsifier_upper_entries.slice(), t_less);
      st.stop(); st.reportTotal("# sort time");

      return sparsifier_upper_entries;
    }

    auto sparsifier_entries = pbbs::sequence<T>(2 * sparsifier_upper_entries.size());

    parallel_for(0, sparsifier_entries.size(), [&] (size_t idx) {
      size_t i = idx >> 1;
      auto kv = sparsifier_upper_entries[i];
      uintE u = (uintE)(std::get<0>(kv) >> 32UL);
      uintE v = (uintE)(std::get<0>(kv));
      double wgh = std::get<1>(kv);
      if ((idx & 1) == 0) {
        // upper triangle, just copy it
        auto pr = (static_cast<size_t>(u)  << 32UL) + static_cast<size_t>(v);
        sparsifier_entries[idx] = std::make_tuple(pr, wgh);
      } else {
        // lower triangle, swap row and column, i.e., (u, v) -> (v, u)
        // treat diagonal entries specially,
        // we set diagonal entries in lower triangle to be -1,
        // since they will finally be removed.
        auto pr = (static_cast<size_t>(v)  << 32UL) + static_cast<size_t>(u);
        sparsifier_entries[idx] = std::make_tuple(pr, (u!=v) ? wgh : -1.0);
      }
    });
    std::cout << "# created sparsifier entries" << std::endl;
    sparsifier_upper_entries.clear();

    auto pred = [&](T& t) { return std::get<1>(t) > 0.0; };

    auto sparsifier_final = pbbslib::filter(sparsifier_entries, pred);
    sparsifier_entries.clear();

    timer st; st.start();
    std::cout << "# starting sort" << std::endl;
    pbbs::sample_sort_inplace(sparsifier_final.slice(), t_less);
    st.stop(); st.reportTotal("# sort time");
    std::cout << "# tot kept = " << sparsifier_final.size() << " in the sparsifier" << std::endl;

    return sparsifier_final;
  }

  template <typename FP>
  FP* compute_u_sigma_root(FP* U, FP* S, size_t n, size_t rank, size_t dim, bool normalize, FP alpha=0.0) {
    std::cout << "# computing u*sqrt(sigma), sigma(0)=" << S[0] << ", sigma(" << dim-1 << ")=" << S[dim-1] << std::endl;
    FP* emb = new FP[n*dim]();
    parallel_for(0, dim, [&] (size_t i) {
      FP filtered_sigma = S[i];
      mklhelper<FP>::cblas_axpy(
          n,                     // const MKL_INT n,
          sqrt(filtered_sigma),  // const float a,
          U+i,                   // const float *x,
          rank,                  // const MKL_INT incx,
          emb+i,                 // float *y,
          dim                    // const MKL_INT incy
      );
    });
    if (normalize) {
      parallel_for(0, n, [&] (size_t i) {
          FP norm = mklhelper<FP>::cblas_nrm2(
              dim,        // const MKL_INT n,
              emb+i*dim,  // const float *x,
              1           // const MKL_INT incx,
          );
          if (norm > EPS) {
              mklhelper<FP>::cblas_scal(
                dim,        // const MKL_INT n,
                1.0/norm,   // const float a,
                emb+i*dim,  // const float *x,
                1           // const MKL_INT incx,
              );
          }
      });
    }
    return emb;
  }

  template <class Graph, typename FP>
  FP* NE_Zhang_et_al(Graph& GA, size_t rank, size_t dim) {
    using W = typename Graph::weight_type;
    auto offs = pbbs::sequence<MKL_INT>(GA.n+1, [&] (size_t i) { return i==GA.n ? 0 : GA.get_vertex(i).getInDegree(); });
    size_t m = pbbslib::scan_add_inplace(offs.slice());
    assert(GA.m == m);
    // MKL_INT* row_idx = new MKL_INT[GA.m];
    MKL_INT* col_idx = new MKL_INT[GA.m];
    FP* value = new FP[GA.m];

    auto negative = pbbs::sequence<double>(GA.n, [&] (size_t j) {
      auto map_f = [&] (const uintE& u, const uintE& v, const W& wgh) {
        // map A_uv to A_uv / D_u
        return static_cast<double>(1.0 / GA.get_vertex(v).getOutDegree());
      };
      // \sigma_i A_ij / D_i
      auto monoid = pbbs::addm<double>();
      double neg =  GA.get_vertex(j).template reduceInNgh<double>(j, map_f, monoid);
      return pow(neg, 0.75);
    });

    double negative_sum = pbbs::reduce(negative, pbbs::addm<double>());

    parallel_for(0, GA.n, [&] (size_t i) {
      size_t k = 0;
      size_t off_i = offs[i];
      auto map_f = [&] (const uintE& u, const uintE& v, const W& wgh) {
        // row_idx[off_i + k] = u;
        col_idx[off_i + k] = v;
        double nominator = 1.0 / GA.get_vertex(u).getOutDegree();
        double denominator = negative[v] / negative_sum;
        value[off_i + k] = static_cast<FP>(log(nominator) - log(denominator));
        ++k;
      };
      GA.get_vertex(i).mapOutNgh(i, map_f, false);
    });
    timer t_svd; t_svd.start();
    MKL_INT* rows_start = offs.to_array();
    MKL_INT* rows_end = rows_start + 1;
    mkl_redsvd::MKLRedSVD<FP> svdOfA(GA.n,
        // row_idx, col_idx, value,
        rows_start, rows_end, col_idx, value,
        false, // upper
        rank,
        true // analyze
        );
    svdOfA.run();
    FP* emb = compute_u_sigma_root<FP>(svdOfA.matU, svdOfA.S, GA.n, rank, dim, true, 0.0);
    t_svd.stop(); t_svd.reportTotal("# svd time");
    return emb;
  }

  template <class Graph>
  void setup_link_prediction(Graph& GA, double ratio, const std::string& edge_file, const std::string& degree_file) {
    auto seed = pbbs::random(0);
    using W = typename Graph::weight_type;
    auto pred = [&](const uintE& u, const uintE& v, const W& wgh) {
      size_t key = (static_cast<size_t>(u) << 32UL) + static_cast<size_t>(v);
      auto our_seed = pbbs::random(seed.ith_rand(key));
      double prob = map_uint64_t(our_seed.rand());
      return prob < ratio;
    };
    auto sampled_edges = sample_edges(GA, pred);
    auto sampled_edges_seq = sampled_edges.to_seq();
    std::cout << "# sampled " << sampled_edges_seq.size() << " edges." << std::endl;
    std::ofstream fedge(edge_file);
    for (size_t i=0; i<sampled_edges_seq.size(); ++i) {
      auto edge = sampled_edges_seq[i];
      uintE u = std::get<0>(edge);
      uintE v = std::get<1>(edge);
      fedge << u << "\t" << v << std::endl;
    }
    auto offs = pbbs::sequence<double>(GA.n, [&] (size_t i) { return static_cast<double>(GA.get_vertex(i).getOutDegree()); });
    double* degrees = offs.to_array();
	  if(std::FILE* fdegree = std::fopen(degree_file.c_str(), "wb")) {
			std::fwrite(degrees, sizeof(double), GA.n, fdegree);
			std::fclose(fdegree);
	  }
    delete[] degrees;
  }

  template <class Graph, typename FP>
  FP* NetSMF(Graph& GA, size_t walks_per_edge, size_t walk_len, bool upper, bool sample, size_t rank, size_t dim, bool analyze, size_t table_size, size_t negative, bool normalize, float sample_ratio, float mem_ratio, const std::vector<float>& step_coeff) {
    timer t_path_sampling; t_path_sampling.start();
    size_t seed = time(0);
    std::cout << "# seed = " << seed << std::endl;
    auto edges = generate_trunc_log_matrix_v2(GA, pbbs::random(seed), walks_per_edge, walk_len, upper, sample, table_size, negative, sample_ratio, mem_ratio, step_coeff);
    t_path_sampling.stop();
    t_path_sampling.reportTotal("# generate matrix time");

    timer t_setup_svd; t_setup_svd.start();
    MKL_INT n = static_cast<MKL_INT>(GA.n);
    size_t num_directed_edges = edges.size();
    MKL_INT nnz = static_cast<MKL_INT>(num_directed_edges);

    // MKL_INT* row_idx = new MKL_INT[num_directed_edges];
    // TODO: consider using mkl_malloc and mkl_free
    // https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/support-functions/memory-management/mkl-malloc.html
    MKL_INT* rows_start = new MKL_INT[GA.n + 1]();
    MKL_INT* col_idx = new MKL_INT[num_directed_edges];
    FP* value = new FP[num_directed_edges];
    parallel_for(0, num_directed_edges, [&] (size_t i) {
      auto kv = edges[i];
      uintE u = (uintE)(std::get<0>(kv) >> 32UL);
      uintE v = (uintE)(std::get<0>(kv));
      V wgh = std::get<1>(kv);
      // row_idx[i] = static_cast<MKL_INT>(u);
      if (i < 10) {
        std::cout << u << " " << v << " " << wgh << std::endl;
      }
      if (i > 0) {
        uintE u_prev = (uintE)(std::get<0>(edges[i-1]) >> 32UL);
        if (u_prev != u) {
          rows_start[u] = i;
        }
      }
      col_idx[i] = static_cast<MKL_INT>(v);
      value[i] = static_cast<FP>(wgh);
    });
    rows_start[GA.n] = num_directed_edges;
    for (size_t i=1; i<=GA.n; ++i) {
      rows_start[i] = std::max(rows_start[i], rows_start[i-1]);
    }
    MKL_INT* rows_end = rows_start + 1;
    edges.clear();
    t_setup_svd.stop(); t_setup_svd.reportTotal("# svd setup time");

    timer t_svd; t_svd.start();
    // mkl_redsvd::MKLRedSVD<FP> svdOfA(n, nnz, row_idx, col_idx, value, upper, rank, analyze);
    mkl_redsvd::MKLRedSVD<FP> svdOfA(n,
        rows_start, rows_end,
        col_idx, value, upper, rank, analyze);
    svdOfA.run();

    FP* emb = compute_u_sigma_root<FP>(svdOfA.matU, svdOfA.S, n, rank, dim, normalize, 0.0);
    t_svd.stop(); t_svd.reportTotal("# svd time");
    return emb;
  }

} // path_embed
