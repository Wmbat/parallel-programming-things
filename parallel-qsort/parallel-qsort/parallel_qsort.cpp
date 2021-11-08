#include <algorithm>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include <mpi.h>

using std::begin;
using std::end;
using std::next;
using std::partition;
using std::prev;

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

auto generate_random_array(i64 count) -> std::vector<i32>;

template <typename It>
auto compute_median_pivot(It begin, It end) -> i32;

template <typename It>
void send_list(It begin, It end, i32 target, MPI_Comm comm);
template <typename It>
auto receive_list(It buffer_begin, i32 target, MPI_Comm comm) -> It;

template <typename It>
auto format_range(It begin, It end) -> std::string;

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

   const i64 elements_per_core = static_cast<i64>(
      std::floor(static_cast<f64>(total_elements) / static_cast<f64>(process_count)));
   std::vector<i32> data_buffer;
   i32 pivot = 0;

   if (process_id == 0)
   {
      data_buffer = generate_random_array(total_elements);
      pivot = compute_median_pivot(begin(data_buffer), end(data_buffer));

      std::cout << "Sorting " << total_elements << " elements\n";
   }
   else
   {
      data_buffer = std::vector<i32>(total_elements, 0);
   }

   auto local_array = std::vector<i32>(elements_per_core);

   MPI_Scatter(static_cast<void*>(data_buffer.data()), static_cast<i32>(elements_per_core),
               MPI_INT32_T, static_cast<void*>(local_array.data()),
               static_cast<i32>(elements_per_core), MPI_INT32_T, 0, MPI_COMM_WORLD);
   MPI_Bcast(&pivot, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

   std::cout << "P" << process_id << " - " << elements_per_core << " random integers received\n";

   auto* communicator = MPI_COMM_WORLD;

   i32 local_rank = process_id;

   const i64 dimensions = static_cast<i64>(std::log2(process_count));
   for (i64 i = dimensions - 1; i >= 0; --i)
   {
      std::cout << "P" << process_id << " - pivot = " << pivot << "\n";
      std::cout << "P" << process_id << " - elements = " << local_array.size() << "\n";

      const auto separator = partition(begin(local_array), end(local_array), [=](i64 v) {
         return v < pivot;
      });

      const i32 target = local_rank ^ (1 << i);
      const i64 local_low_list_size = std::distance(begin(local_array), separator);
      const i64 local_high_list_size = std::distance(separator, end(local_array));

      if ((process_id & (1 << i)) == 0)
      {
         std::cout << "P" << process_id << " - sending high-list\n";

         send_list(separator, end(local_array), target, communicator);

         std::cout << "P" << process_id << " - receiving low-list\n";

         const auto recv_end = receive_list(begin(data_buffer), target, communicator);

         std::cout << "P" << process_id << " - merging\n";

         const i64 recv_size = std::distance(begin(data_buffer), recv_end);

         local_array.resize(local_low_list_size + recv_size);
         std::copy(begin(data_buffer), recv_end, begin(local_array) + local_low_list_size);
      }
      else
      {
         std::cout << "P" << process_id << " - receiving high-list\n";

         const auto recv_end = receive_list(begin(data_buffer), target, communicator);

         std::cout << "P" << process_id << " - sending low-list\n";

         send_list(begin(local_array), separator, target, communicator);

         std::cout << "P" << process_id << " - merging\n";

         const i64 recv_size = std::distance(begin(data_buffer), recv_end);

         std::rotate(begin(local_array), begin(local_array) + local_low_list_size,
                     end(local_array));
         local_array.resize(local_high_list_size + recv_size);
         std::copy(begin(data_buffer), recv_end, begin(local_array) + local_high_list_size);
      }

      if (i >= 0)
      {
         MPI_Comm_split(communicator, local_rank & (1 << i), process_id, &communicator);
         MPI_Comm_rank(communicator, &local_rank);

         if (local_rank == 0)
         {
            pivot = compute_median_pivot(begin(local_array), end(local_array));
         }

         MPI_Bcast(&pivot, 1, MPI_INT32_T, 0, communicator);
      }
   }

   std::cout << "P" << process_id << " - performing local quicksort\n";

   qsort(begin(local_array), end(local_array));

   i32 local_size = static_cast<i32>(local_array.size());
   auto sizes = std::vector<i32>(process_count, 0);
   auto displacements = std::vector<i32>(process_count, 0);

   MPI_Gather(&local_size, 1, MPI_INT32_T, sizes.data(), 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

   std::partial_sum(begin(sizes), prev(end(sizes)), next(begin(displacements)));

   MPI_Gatherv(local_array.data(), static_cast<i32>(local_array.size()), MPI_INT32_T,
               data_buffer.data(), sizes.data(), displacements.data(), MPI_INT32_T, 0,
               MPI_COMM_WORLD);

   const f64 elapsed_time = MPI_Wtime() - start_time;
   MPI_Finalize();

   if (process_id == 0)
   {
      i64 true_size = elements_per_core * process_count;
      std::cout << "data: {" << format_range(begin(data_buffer), begin(data_buffer) + true_size)
                << "}\n";
      std::cout << "elapsed time: " << elapsed_time << '\n';
   }

   return 0;
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

template <typename It>
auto compute_median_pivot(It begin, It end) -> i32
{
   const i32 size = std::distance(begin, end);
   const i32 sum = std::accumulate(begin, end, 0);
   const float median = static_cast<float>(sum) / static_cast<float>(size);

   return static_cast<i32>(std::ceil(median));
}

template <typename It>
void send_list(It begin, It end, i32 target, MPI_Comm comm)
{
   i32 size = std::distance(begin, end);

   MPI_Send(&size, 1, MPI_INT32_T, target, 0, comm);
   MPI_Send(begin.base(), size, MPI_INT32_T, target, 0, comm);
}
template <typename It>
auto receive_list(It buffer_begin, i32 target, MPI_Comm comm) -> It
{
   i32 recv_size = 0;
   MPI_Recv(&recv_size, 1, MPI_INT32_T, target, 0, comm, nullptr);

   MPI_Recv(buffer_begin.base(), recv_size, MPI_INT32_T, target, 0, comm, nullptr);

   return buffer_begin + recv_size;
}

template <typename It>
auto format_range(It begin, It end) -> std::string
{
   std::string str = " ";
   std::for_each(begin, end, [&](const auto& v) {
      str += std::to_string(v);
      str += " ";
   });

   return str;
}

auto is_power_of_2(i32 n) -> bool
{
   return n && !(n & (n - 1));
}
