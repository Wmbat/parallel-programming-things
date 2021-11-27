#ifndef SEQUENTIAL_FLOYD_WARSHALL_GRAPH_GRAPH_HPP_
#define SEQUENTIAL_FLOYD_WARSHALL_GRAPH_GRAPH_HPP_

#include <sequential-floyd-warshall/graph/edge.hpp>
#include <sequential-floyd-warshall/graph/node.hpp>

#include <vector>

class graph
{
public:
   using size_type = std::size_t;
   using iterator = typename std::vector<node>::iterator;
   using const_iterator = typename std::vector<node>::const_iterator;
   using reverse_iterator = typename std::vector<node>::reverse_iterator;
   using const_reverse_iterator = typename std::vector<node>::const_reverse_iterator;

public:
   void add_connection(u32 index, edge e);

   auto size() const noexcept -> const size_type;

   auto begin() noexcept -> iterator;
   auto begin() const noexcept -> const_iterator;
   auto cbegin() const noexcept -> const_iterator;

   auto end() noexcept -> iterator;
   auto end() const noexcept -> const_iterator;
   auto cend() const noexcept -> const_iterator;

private:
   std::vector<node> m_adjacency_list;
};

#endif // SEQUENTIAL_FLOYD_WARSHALL_GRAPH_GRAPH_HPP_
