#include <parallel-floyd-warshall/graph/graph.hpp>
#include <parallel-floyd-warshall/types.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <mpi.h>

static constexpr i32 tombstone = std::numeric_limits<i32>::max();

void print(const graph& g)
{
   for (const auto& node : g)
   {
      for (const auto edge : node.edges)
      {
         std::cout << node.index << " |-- " << edge.weight << " --> " << edge.end << "\n";
      }

      if (node.edges.empty())
      {
         std::cout << node.index << "\n";
      }
   }
}

auto create_adjacency_matrix(const graph& g) -> std::vector<i32>
{
   auto dist = std::vector<i32>(g.size() * g.size(), tombstone);

   for (const auto& n : g)
   {
      dist[n.index * g.size() + n.index] = 0;
   }

   for (const auto& n : g)
   {
      for (const auto& e : n.edges)
      {
         dist[n.index * g.size() + e.end] = e.weight;
      }
   }

   return dist;
}
auto reorganize_matrix(const std::vector<i32>& m, i32 div_count) -> std::vector<i32>
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

auto to_string(const std::vector<i32>& m) -> std::string
{
   const u64 width = static_cast<u64>(std::sqrt(m.size()));

   std::string str;
   str.reserve(m.size());

   int i = 0;
   for (i32 val : m)
   {
      if (i == width)
      {
         str += "\n";
         i = 0;
      }

      if (val == tombstone)
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

   return str;
}

template <typename It>
auto format_range(It begin, It end) -> std::string;

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
      graph g;
      g.add_connection(0, edge{-2, 2});
      g.add_connection(1, edge{4, 0});
      g.add_connection(1, edge{3, 2});
      g.add_connection(2, edge{2, 3});
      g.add_connection(3, edge{-1, 1});

      const auto matrix = create_adjacency_matrix(g);

      std::cout << "P" << process_id << " - Original Matrix\n";
      std::cout << to_string(matrix) << "\n";

      interlaced_matrix = reorganize_matrix(matrix, process_count);

      std::cout << "P0 - adjacency matrix size = " << matrix.size() << "\n";
      std::cout << "P0 - interlaced matrix size = " << interlaced_matrix.size() << "\n";
      std::cout << "P0 - Scattering matrix\n";
   }

   i32 local_size = i32(interlaced_matrix.size()) / process_count;
   MPI_Bcast(&local_size, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

   auto local_matrix = std::vector<i32>(local_size, 0);
   MPI_Scatter(interlaced_matrix.data(), local_size, MPI_INT32_T, local_matrix.data(), local_size,
               MPI_INT32_T, 0, MPI_COMM_WORLD);

   std::cout << "P" << process_id << " - local matrix:\n" << to_string(local_matrix) << "\n";

   i32 col_color = process_id % static_cast<i32>(std::sqrt(process_count));
   i32 row_color = std::floor(static_cast<f32>(process_id) / std::sqrt(process_count));

   std::cout << "P" << process_id << " - row colour = " << row_color << "\n";
   std::cout << "P" << process_id << " - column colour = " << col_color << "\n";

   MPI_Comm col_comm = {};
   MPI_Comm row_comm = {};
   MPI_Comm_split(MPI_COMM_WORLD, col_color, process_id, &col_comm);
   MPI_Comm_split(MPI_COMM_WORLD, row_color, process_id, &row_comm);

   i32 local_width = static_cast<i32>(std::sqrt(local_size));
   i32 total_width = static_cast<i32>(std::sqrt(interlaced_matrix.size()));
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
            if (kth_row[j] != tombstone and kth_col[i] != tombstone)
            {
               local_matrix[i + j * local_width] =
                  std::min(kth_col[i] + kth_row[j], local_matrix[i + j * local_width]);
            }
         }
      }

      std::cout << "P" << process_id << " - local matrix:\n" << to_string(local_matrix) << "\n";
   }

   MPI_Gather(local_matrix.data(), local_matrix.size(), MPI_INT32_T, interlaced_matrix.data(),
              local_matrix.size(), MPI_INT32_T, 0, MPI_COMM_WORLD);

   const f64 elapsed_time = MPI_Wtime() - start_time;
   MPI_Finalize();

   if (process_id == 0)
   {
      std::cout << "data: {" << format_range(begin(interlaced_matrix), end(interlaced_matrix))
                << "}\n";
      std::cout << "elapsed time: " << elapsed_time << '\n';
   }

   return 0;
}

template <typename It>
auto format_range(It begin, It end) -> std::string
{
   using type = typename It::value_type;

   std::string str = " ";
   std::for_each(begin, end, [&](const type& val) {
      if (val == tombstone)
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

auto is_power_of_2(i32 n) -> bool
{
   return n && !(n & (n - 1));
}
