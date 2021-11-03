#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include <mpi.h>

using std::begin;
using std::end;
using std::partition;

using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using f64 = double;

using iterator = typename std::vector<i64>::iterator;

static constexpr i64 random_generation_bound = 1000;
static constexpr i64 element_per_core = 64;

auto generate_random_array(i64 count) -> std::vector<i64>
{
   std::random_device rd;
   auto random_engine = std::default_random_engine(rd());
   auto distribution = std::uniform_int_distribution<i64>(0, random_generation_bound);

   auto data = std::vector<i64>(count);
   for (auto& val : data)
   {
      val = distribution(random_engine);
   }

   return data;
}

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

void send_list(iterator begin, iterator end, i32 target, MPI_Comm comm);
auto receive_list(i32 target, MPI_Comm comm) -> std::vector<i64>;

auto format_vector(const std::vector<i64>& data) -> std::string;

auto is_power_of_2(i32 n) -> bool;

auto main(int argc, char* argv[]) -> int
{
   int process_id = 0;
   int process_count = 0;

   MPI_Init(&argc, &argv);

   const f64 start_time = MPI_Wtime();

   MPI_Comm_size(MPI_COMM_WORLD, &process_count);
   MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

   if (not is_power_of_2(process_count))
   {
      std::cout << "Process count (" << process_count << ") is not a power of 2\n";

      return EXIT_FAILURE;
   }

   std ::vector<i64> original_data;
   i64 pivot = 0;

   if (process_id == 0)
   {
      original_data = generate_random_array(element_per_core * process_count);

      const i64 sum = std::accumulate(begin(original_data), end(original_data), 0L);
      const f64 median = static_cast<double>(sum) / static_cast<double>(original_data.size());
      pivot = static_cast<i64>(std::ceil(median));
   }

   auto local_array = std::vector<i64>(element_per_core);

   MPI_Scatter(static_cast<void*>(original_data.data()), static_cast<i32>(element_per_core),
               MPI_INT64_T, static_cast<void*>(local_array.data()),
               static_cast<i32>(element_per_core), MPI_INT64_T, 0, MPI_COMM_WORLD);
   MPI_Bcast(&pivot, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

   auto* communicator = MPI_COMM_WORLD;

   const i64 dimensions = static_cast<i64>(std::log2(process_count));
   for (i64 i = dimensions - 1; i >= 0; --i)
   {
      if (i >= 0)
      {
         MPI_Comm_split(communicator, process_id & (1 << i), process_id, &communicator);
         MPI_Comm_rank(communicator, &process_id);

         if (process_id == 0)
         {
            const i64 sum = std::accumulate(begin(local_array), end(local_array), 0L);
            const f64 median = static_cast<double>(sum) / static_cast<double>(local_array.size());
            pivot = static_cast<i64>(std::ceil(median));
         }

         MPI_Bcast(&pivot, 1, MPI_INT64_T, 0, communicator);
      }

      std::cout << "P" << process_id << " pivot = " << pivot << "\n";
      std::cout << "P" << process_id << " elements = " << local_array.size() << "\n";

      const auto separator = partition(begin(local_array), end(local_array), [=](i64 v) {
         return v < pivot;
      });

      const i32 target = process_id ^ (1 << i);
      const i64 local_low_list_size = std::distance(begin(local_array), separator);
      const i64 local_high_list_size = std::distance(separator, end(local_array));

      if ((process_id & (1 << i)) == 0)
      {
         std::cout << "P" << process_id << " sending high-list\n";

         send_list(separator, end(local_array), target, communicator);

         std::cout << "P" << process_id << " receiving low-list\n";

         const auto recv_low_list = receive_list(target, communicator);

         std::cout << "P" << process_id << " merging\n";

         auto new_array = std::vector<i64>(recv_low_list.size() + local_low_list_size);
         std::copy(begin(local_array), separator, begin(new_array));
         std::copy(begin(recv_low_list), end(recv_low_list),
                   begin(new_array) + local_low_list_size);

         local_array = new_array;
      }
      else
      {
         std::cout << "P" << process_id << " sending low-list\n";

         send_list(begin(local_array), separator, target, communicator);

         std::cout << "P" << process_id << " receiving high-list\n";

         const auto recv_high_list = receive_list(target, communicator);

         std::cout << "P" << process_id << " merging\n";

         auto new_array = std::vector<i64>(local_high_list_size + recv_high_list.size());
         std::copy(separator, end(local_array), begin(new_array));
         std::copy(begin(recv_high_list), end(recv_high_list),
                   begin(new_array) + local_high_list_size);

         local_array = new_array;
      }
   }

   std::cout << "P" << process_id << " performing local quicksort\n";

   qsort(begin(local_array), end(local_array));

   std::cout << "P" << process_id << " sorted array = {" << format_vector(local_array) << "}\n";

   MPI_Finalize();

   return 0;
}

void send_list(iterator begin, iterator end, i32 target, MPI_Comm comm)
{
   i64 size = std::distance(begin, end);

   MPI_Send(&size, 1, MPI_INT64_T, target, 0, comm);
   MPI_Send(begin.base(), static_cast<i32>(size), MPI_INT64_T, target, 0, comm);
}
auto receive_list(i32 target, MPI_Comm comm) -> std::vector<i64>
{
   i64 recv_size = 0;
   MPI_Recv(&recv_size, 1, MPI_INT64_T, target, 0, comm, nullptr);

   auto array = std::vector<i64>(recv_size);
   MPI_Recv(array.data(), static_cast<i32>(recv_size), MPI_INT64_T, target, 0, comm, nullptr);

   return array;
}

auto format_vector(const std::vector<i64>& data) -> std::string
{
   std::string str = " ";
   for (auto i : data)
   {
      str += std::to_string(i);
      str += " ";
   }

   return str;
}

auto is_power_of_2(i32 n) -> bool
{
   return n && !(n & (n - 1));
}
