#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <vector>

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

auto main() -> int
{
   auto test = std::vector<int>({3, 5, 8, 13, 2, 1}); // NOLINT

   qsort(test.begin(), test.end());

   for (auto i : test)
   {
      std::cout << i << '\n';
   }

   return 0;
}
