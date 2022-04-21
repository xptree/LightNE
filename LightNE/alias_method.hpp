#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <limits>
#include <functional>
#include <map>
#include <vector>
#include <queue>

class discrete_random_variable {
 private:
  const std::vector<std::pair<float, size_t>> alias_;

 public:
  discrete_random_variable(const std::vector<float>& probs) : alias_(generate_alias_table(probs)) {
    const float sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    assert(std::fabs(1.0 - sum) < std::numeric_limits<float>::epsilon());
  }

  size_t sample(size_t idx, double real_dis) const {
    if (real_dis >= alias_[idx].first and
          alias_[idx].second != std::numeric_limits<size_t>::max()) {
      return alias_[idx].second;
    } else {
      return idx;
    }
  }

 private:
  std::vector<std::pair<float, size_t>> generate_alias_table(const std::vector<float>& probs) {
    const size_t sz = probs.size();
    std::vector<std::pair<float, size_t>> alias(sz, {0.0, std::numeric_limits<size_t>::max()});
    std::queue<size_t>  small, large;

    for (size_t i = 0; i != sz; ++i) {
      alias[i].first = sz * probs[i];
      if (alias[i].first < 1.0) {
        small.push(i);
      } else {
        large.push(i);
      }
    }

    while (not(small.empty()) and not(large.empty())) {
      auto s = small.front(), l = large.front();
      small.pop(), large.pop();
      alias[s].second = l;
      alias[l].first -= (1.0 - alias[s].first);

      if (alias[l].first < 1.0) {
        small.push(l);
      } else {
        large.push(l);
      }
    }

    return alias;
  }
};

