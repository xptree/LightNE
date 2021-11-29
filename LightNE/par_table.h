#pragma once

#include <tuple>
#include <cassert>

#include "ligra/bridge.h"

template <class K, class V, class H>
class par_table {
 public:
  using T = std::tuple<K, V>;

  size_t m;
  size_t mask;
  T empty;
  K empty_key;
  T* table;
  bool alloc;
  H& hash_f;

  size_t size() {
    return m;
  }

  void copy_table(T* table2){
    std::cout<<"m:"<<m<<std::endl;
    parallel_for(0, m, [&] (size_t i) {
      table[i] = table2[i];
    });
  }

  static void clearA(T* A, long n, T kv) {
    par_for(0, n, pbbslib::kSequentialForThreshold, [&] (size_t i)
                    { A[i] = kv; });
  }

  inline size_t hashToRange(size_t h) { return h & mask; }
  inline size_t firstIndex(K& k) { return hashToRange(hash_f(k)); }
  inline size_t incrementIndex(size_t h) { return hashToRange(h + 1); }
  inline size_t incrementIndexBy(size_t h, size_t step) { return hashToRange(h + step); }

  void del() {
    if (alloc) {
      cout << "freed hash table" << endl;
      pbbslib::free_array(table);
      alloc = false;
    }
  }

  // gives up ownership of space
  T* to_array() {
    T* r = table;  
    table = NULL; 
    m = 0;
    alloc = false; 
    return r;
  }
 

  par_table() : m(0) {
    mask = 0;
    alloc = false;
  }

  // Size is the maximum number of values the hash table will hold.
  // Overfilling the table could put it into an infinite loop.
  par_table(size_t _m, T _empty, H& hash_f, long space_mult=-1)
      : empty(_empty),
        empty_key(std::get<0>(empty)),
        hash_f(hash_f) {
    if (space_mult == -1) space_mult = 1.1;
    m = (size_t)1 << pbbslib::log2_up((size_t)(space_mult * _m));
    mask = m - 1;
    table = pbbslib::new_array_no_init<T>(m);
    clearA(table, m, empty);
    alloc = true;
  }

//  // Size is the maximum number of values the hash table will hold.
//  // Overfilling the table could put it into an infinite loop.
//  par_table(size_t _m, T _empty, T* _tab, bool clear=true)
//      : m(_m),
//        mask(m - 1),
//        table(_tab),
//        empty(_empty),
//        empty_key(std::get<0>(empty)) {
//    if (clear) {
//      clearA(table, m, empty);
//    }
//    alloc = false;
//  }

  void insert_add(std::tuple<K, V> kv) {
    K k = std::get<0>(kv);
    size_t ctr = 0;
    size_t h = firstIndex(k);
    while (true) {
      if (std::get<0>(table[h]) == empty_key) {
        if (pbbslib::atomic_compare_and_swap(&table[h], empty, kv)) {
          return;
        }
      }
      if (std::get<0>(table[h]) == k) {
//        (*(&std::get<1>(table[h])))++;
        pbbslib::fetch_and_add(&std::get<1>(table[h]), std::get<1>(kv));
        return;
      }
      ctr++;
      // h = incrementIndexBy(h, ctr);
      h = incrementIndex(h);
      if (ctr > 10000000) {
        std::cout << "looping forever" << std::endl;
        exit(0);
      }
    }
  }

  template <class F>
  bool insert_f(std::tuple<K, V> kv, const F& f) {
    K k = std::get<0>(kv);
    size_t h = firstIndex(k);
    while (true) {
      if (std::get<0>(table[h]) == empty_key) {
        if (pbbslib::CAS(&std::get<0>(table[h]), empty_key, k)) {
          std::get<1>(table[h]) = std::get<1>(kv);
          return true;
        }
      }
      if (std::get<0>(table[h]) == k) {
        f(&std::get<1>(table[h]), kv);
        return false;
      }
      h = incrementIndex(h);
    }
    return false;
  }

  bool contains(K k) {
    size_t h = firstIndex(k);
    while (true) {
      if (std::get<0>(table[h]) == k) {
        return true;
      } else if (std::get<0>(table[h]) == empty_key) {
        return false;
      }
      h = incrementIndex(h);
    }
    return false;
  }

  V find(K k, V default_value) {
    size_t h = firstIndex(k);
    while (true) {
      if (std::get<0>(table[h]) == k) {
        return std::get<1>(table[h]);
      } else if (std::get<0>(table[h]) == empty_key) {
        return default_value;
      }
      h = incrementIndex(h);
    }
    return default_value;
  }

  sequence<T> entries() {
    auto pred = [&](T& t) { return std::get<0>(t) != empty_key; };
    auto table_seq = pbbslib::make_sequence<T>(table, m);
    return pbbslib::filter(table_seq, pred);
  }

  template <class M>
  void map_table(M& map_f) {
    parallel_for(0, m, [&] (size_t i) {
      if (std::get<0>(table[i]) != empty_key) {
        table[i] = map_f(table[i]);
      }
    });
  }

  template <class P>
  sequence<T> predicated_entries(P& p) {
    auto pred = [&](T& t) {
      if (std::get<0>(t) != empty_key) {
        return p(t);
      }
      return false;
    };
    auto table_seq = pbbslib::make_sequence<T>(table, m);
    return pbbslib::filter(table_seq, pred);
  }

  void clear() {
    par_for(0, m, 2048, [&] (size_t i) { table[i] = empty; });
  }
};

template <class K, class V, class F>
auto make_par_table(size_t _m, std::tuple<K, V> _empty, F& hash_f, long space_mult=-1) {
  return par_table<K, V, F>(_m, _empty, hash_f, space_mult);
}
