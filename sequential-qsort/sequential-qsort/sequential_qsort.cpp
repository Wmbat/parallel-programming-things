#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using f64 = double;

static constexpr i64 random_generation_bound = 1000;
static constexpr i64 total_elements = 10000;

template <typename It>
void qsort(It beg, It end)
{
   using value_type = typename std::iterator_traits<It>::value_type;

   if (beg < end)
   {
      const auto pivot = std::prev(end);
      const auto separator = std::partition(beg, std::prev(end), [pivot](const value_type& v) {
         return v < *pivot;
      });

      std::iter_swap(separator, pivot);

      qsort(beg, separator);
      qsort(std::next(separator), end);
   }
}

auto generate_random_array(i64 count) -> std::vector<i32>
{
   std::random_device rd;
   auto random_engine = std::default_random_engine(rd());
   auto distribution = std::uniform_int_distribution<i32>(0, random_generation_bound);

   auto data = std::vector<i32>(count);
   for (auto& val : data)
   {
      val = distribution(random_engine);
   }

   return data;
}

auto main() -> int
{
   auto test = generate_random_array(total_elements);

   qsort(test.begin(), test.end());

   for (auto i : test)
   {
      std::cout << i << '\n';
   }

   return 0;
}
