#include <parallel-floyd-warshall/graph/graph.hpp>

#include <algorithm>
#include <iostream>

void graph::add_connection(u32 index, edge e)
{
   const auto it =
      std::find_if(std::begin(m_adjacency_list), std::end(m_adjacency_list), [=](const node& n) {
         return n.index == index;
      });

   if (it != std::end(m_adjacency_list))
   { // The node is already in the list
      it->edges.push_back(e);

      const auto edge_it =
         std::find_if(std::begin(m_adjacency_list), std::end(m_adjacency_list), [&](const node& n) {
            return n.index == e.end;
         });

      if (it == std::end(m_adjacency_list)) {
         m_adjacency_list.push_back(node{e.end});
      }
   }
   else
   {
      m_adjacency_list.push_back(node{index, {e}});
   }
}

auto graph::size() const noexcept -> const size_type
{
   return m_adjacency_list.size();
}

auto graph::begin() noexcept -> iterator
{
   return m_adjacency_list.begin();
}
auto graph::begin() const noexcept -> const_iterator
{
   return m_adjacency_list.begin();
}
auto graph::cbegin() const noexcept -> const_iterator
{
   return m_adjacency_list.cbegin();
}

auto graph::end() noexcept -> iterator
{
   return m_adjacency_list.end();
}
auto graph::end() const noexcept -> const_iterator
{
   return m_adjacency_list.end();
}
auto graph::cend() const noexcept -> const_iterator
{
   return m_adjacency_list.cend();
}
