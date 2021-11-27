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

   auto interlaced = std::vector<i32>(m.size());
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

   std::string ret;
   ret.reserve(m.size());

   int i = 0;
   for (i32 val : m)
   {
      if (i == width)
      {
         ret += "\n";
         i = 0;
      }

      if (val == tombstone)
      {
         ret += "_ ";
      }
      else
      {
         ret += std::to_string(val);
         ret += " ";
      }

      ++i;
   }
}

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

      std::cout << "GRAPH\n";
      print(g);

      std::cout << "P" << process_id << " - Original Matrix\n";

      const auto matrix = create_adjacency_matrix(g);
      interlaced_matrix = reorganize_matrix(matrix, process_count);
   }

   const i32 per_matrix_count = i32(interlaced_matrix.size()) / process_count;
   auto local_matrix = std::vector<i32>(per_matrix_count, 0);
   MPI_Scatter(interlaced_matrix.data(), per_matrix_count, MPI_INT32_T, local_matrix.data(),
               per_matrix_count, MPI_INT32_T, 0, MPI_COMM_WORLD);

   const auto str = to_string(local_matrix);

   std::cout << "\nP" << process_id << " - matrix:\n" << str;

   /*
   for (const auto& row : dist)
   {
      for (auto j : row)
      {
         if (j == tombstone)
         {
            std::cout << "_ ";
         }
         else
         {
            std::cout << j << " ";
         }
      }

      std::cout << "\n";
   }
   */

   return 0;
}

auto is_power_of_2(i32 n) -> bool
{
   return n && !(n & (n - 1));
}
