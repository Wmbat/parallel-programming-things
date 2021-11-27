#include <algorithm>
#include <sequential-floyd-warshall/graph/graph.hpp>
#include <sequential-floyd-warshall/types.hpp>

#include <iostream>
#include <limits>

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

auto create_adjacency_matrix(const graph& g) -> std::vector<std::vector<i32>>
{
   auto dist = std::vector<std::vector<i32>>(g.size(), std::vector<i32>(g.size(), tombstone));

   for (const auto& n : g)
   {
      dist[n.index][n.index] = 0;
   }

   for (const auto& n : g)
   {
      for (const auto& e : n.edges)
      {
         dist[n.index][e.end] = e.weight;
      }
   }

   return dist;
}

auto main() -> int
{
   graph g;
   g.add_connection(0, edge{-2, 2});
   g.add_connection(1, edge{4, 0});
   g.add_connection(1, edge{3, 2});
   g.add_connection(2, edge{2, 3});
   g.add_connection(3, edge{-1, 1});

   print(g);

   auto dist = create_adjacency_matrix(g);

   std::cout << "\nMATRIX\n";

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

   for (int k = 0; k < g.size(); ++k)
   {
      for (int j = 0; j < g.size(); ++j)
      {
         for (int i = 0; i < g.size(); ++i)
         {
            if (dist[i][k] != tombstone and dist[k][j] != tombstone)
            {
               dist[i][j] = std::min(dist[i][k] + dist[k][j], dist[i][j]);
            }
         }
      }
   }

   std::cout << "\nAfter floyd-warshall\n";

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

   return 0;
}
