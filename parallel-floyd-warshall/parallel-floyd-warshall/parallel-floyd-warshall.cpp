#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include <mpi.h>

using i32 = std::int32_t;
using i64 = std::int64_t;

using u32 = std::uint32_t;
using u64 = std::uint64_t;

using f32 = float;
using f64 = double;

static constexpr i32 mark = std::numeric_limits<i32>::max();

static const auto matrix = std::vector<i32>(
   {0,    1,    mark, mark, 4,    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, 0,    2,    mark, 3,    mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, 0,    3,    2,    mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, 0,
    4,    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, 0,    5,    2,    mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, 20,   mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, 6,    mark, mark, mark, mark, 0,    mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, 0,    1,
    mark, mark, mark, mark, 6,    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, 0,    2,    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, 0,    3,    2,    mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, 0,    4,    mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, 0,    5,    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, 6,    mark, 3,    mark, mark, 0,    1,    mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, 0,    1,    mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, 0,    2,    mark, 3,    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, 0,    3,    mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, 0,    4,    mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    0,    5,    mark, mark, mark, mark, mark, 10,   mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    6,    mark, mark, mark, mark, 0,    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, 0,    mark, 2,    mark, mark, 6,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, 5,    0,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, 4,    0,    mark, 2,    mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, 3,    0,    mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, 14,   mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, 2,    0,    mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, 2,    1,    0,    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, 0,    1,    mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, 25,   mark, mark, 10,   mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, 0,    2,    mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, 0,    1,    2,    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, 0,    3,    mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, 4,    3,    mark, mark, 0,    mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, 0,    mark, mark, mark, mark, 6,    mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, 2,    5,    0,    mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, 4,    0,    mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, 2,    3,
    0,    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, 2,    0,    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, 3,    mark, 1,    0,    mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark,
    mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, mark, 1,    0});

auto interlace_matrix(const std::vector<i32>& m, i32 div_count) -> std::vector<i32>;
auto deinterlace_matrix(const std::vector<i32>& m, i32 div_count) -> std::vector<i32>;

template <typename It>
auto format_range(It begin, It end) -> std::string;
auto format_matrix(const std::vector<i32>& matrix) -> std::string;

auto is_power_of_2(i32 n) -> bool;

auto main(int argc, char** argv) -> int
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

   std::vector<i32> interlaced_matrix;
   if (process_id == 0)
   {
      interlaced_matrix = interlace_matrix(matrix, process_count);

      std::cout << "P0 - Scattering matrix\n";
   }

   i32 total_size = i32(interlaced_matrix.size());
   MPI_Bcast(&total_size, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

   i32 local_size = total_size / process_count;

   auto local_matrix = std::vector<i32>(local_size, 0);
   MPI_Scatter(interlaced_matrix.data(), local_size, MPI_INT32_T, local_matrix.data(), local_size,
               MPI_INT32_T, 0, MPI_COMM_WORLD);

   std::cout << "P" << process_id << " - local matrix:\n" << format_matrix(local_matrix) << "\n";

   i32 col_color = process_id % static_cast<i32>(std::sqrt(process_count));
   i32 row_color = std::floor(static_cast<f32>(process_id) / std::sqrt(process_count));

   std::cout << "P" << process_id << " - row colour = " << row_color << "\n";
   std::cout << "P" << process_id << " - column colour = " << col_color << "\n";

   MPI_Comm col_comm = {};
   MPI_Comm row_comm = {};
   MPI_Comm_split(MPI_COMM_WORLD, col_color, process_id, &col_comm);
   MPI_Comm_split(MPI_COMM_WORLD, row_color, process_id, &row_comm);

   i32 local_width = static_cast<i32>(std::sqrt(local_size));
   i32 total_width = static_cast<i32>(std::sqrt(total_size));
   for (int k = 0; k < total_width; ++k)
   {
      const auto k_process_index = k / local_width;

      auto kth_col = std::vector<i32>(local_width);
      if (k_process_index == col_color)
      {
         const u32 col_offset = k % local_width;
         for (u32 i = 0; i < kth_col.size(); ++i)
         {
            kth_col[i] = local_matrix[col_offset + local_width * i];
         }

         std::cout << "P" << process_id << " - broadcasting " << k << "th column to rows\n";
      }

      MPI_Bcast(kth_col.data(), local_width, MPI_INT32_T, k_process_index, row_comm);

      auto kth_row = std::vector<i32>(local_width);
      if (k_process_index == row_color)
      {
         const u32 row_offset = k % local_width;
         for (int i = 0; i < kth_row.size(); ++i)
         {
            kth_row[i] = local_matrix[row_offset * local_width + i];
         }

         std::cout << "P" << process_id << " - broadcasting " << k << "th row to columns\n";
      }

      MPI_Bcast(kth_row.data(), local_width, MPI_INT32_T, k_process_index, col_comm);
      std::cout << "P" << process_id << " - Receiving " << k
                << "th row: " << format_range(std::begin(kth_row), std::end(kth_row)) << "\n";

      for (int j = 0; j < local_width; ++j)
      {
         for (int i = 0; i < local_width; ++i)
         {
            if (kth_row[j] != mark and kth_col[i] != mark)
            {
               const i32 offset = j + i * local_width;
               local_matrix[offset] = std::min(kth_col[i] + kth_row[j], local_matrix[offset]);
            }
         }
      }

      std::cout << "P" << process_id << " - local matrix:\n" << format_matrix(local_matrix) << "\n";
   }

   MPI_Gather(local_matrix.data(), static_cast<i32>(local_matrix.size()), MPI_INT32_T,
              interlaced_matrix.data(), static_cast<i32>(local_matrix.size()), MPI_INT32_T, 0,
              MPI_COMM_WORLD);

   if (process_id == 0)
   {
      std::cout << "\n\n"
                << format_matrix(deinterlace_matrix(interlaced_matrix, process_count)) << "\n\n";
   }

   MPI_Finalize();

   return 0;
}

auto interlace_matrix(const std::vector<i32>& m, i32 div_count) -> std::vector<i32>
{
   const u64 total_size = static_cast<u64>(std::sqrt(m.size()));
   const u64 process_row_count = static_cast<u64>(std::sqrt(div_count));
   const u64 divided_size = static_cast<u64>(static_cast<f64>(total_size) / std::sqrt(div_count));

   auto matrices = std::vector<std::vector<i32>>(div_count);

   for (u64 i = 0u; i < total_size; ++i)
   {
      for (u64 j = 0u; j < total_size; ++j)
      {
         const auto row = i / divided_size;
         const auto col = j / divided_size;

         auto& mat = matrices[row * process_row_count + col];
         mat.push_back(m[i * total_size + j]);
      }
   }

   auto interlaced = std::vector<i32>();
   interlaced.reserve(m.size());
   for (const auto& mat : matrices)
   {
      for (i32 val : mat)
      {
         interlaced.push_back(val);
      }
   }

   return interlaced;
}
auto deinterlace_matrix(const std::vector<i32>& m, i32 process_count) -> std::vector<i32>
{
   const u64 total_matrix_size = m.size();
   const u64 local_matrix_size = static_cast<u64>(std::sqrt(m.size()));
   const u64 local_matrix_width = static_cast<u64>(std::sqrt(local_matrix_size));
   const u64 processes_per_row = static_cast<u64>(std::sqrt(process_count));
   const u64 size_by_p_row = local_matrix_size / processes_per_row;

   std::vector<i32> result = std::vector<i32>(total_matrix_size);
   for (u64 node = 0; node < process_count; node++)
   {
      const auto row_offset = node / processes_per_row;
      const auto col_offset = node % processes_per_row;

      for (int i = 0; i < size_by_p_row; ++i)
      {
         for (int j = 0; j < size_by_p_row; ++j)
         {
            result[row_offset * size_by_p_row * local_matrix_size + col_offset * size_by_p_row +
                   i * local_matrix_size + j] =
               m[node * size_by_p_row * size_by_p_row + i * size_by_p_row + j];
         }
      }
   }

   return result;
}

template <typename It>
auto format_range(It begin, It end) -> std::string
{
   using type = typename It::value_type;

   std::string str = " ";
   std::for_each(begin, end, [&](const type& val) {
      if (val == mark)
      {
         str += "_ ";
      }
      else
      {
         str += std::to_string(val);
         str += " ";
      }
   });

   return str;
}
auto format_matrix(const std::vector<i32>& matrix) -> std::string
{
   const auto width = static_cast<i32>(std::sqrt(matrix.size()));

   std::string str;
   i32 i = 0;
   for (i32 val : matrix)
   {
      if (i == width)
      {
         str += "\n";
         i = 0;
      }

      if (val == mark)
      {
         str += "_ ";
      }
      else
      {
         str += std::to_string(val);
         str += " ";
      }
      ++i;
   }

   return str.substr(0, str.size() - 2);
}

auto is_power_of_2(i32 n) -> bool
{
   return n && !(n & (n - 1));
}
