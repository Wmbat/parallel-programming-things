#ifndef SEQUENTIAL_FLOYD_WARSHALL_GRAPH_NODE_HPP_
#define SEQUENTIAL_FLOYD_WARSHALL_GRAPH_NODE_HPP_

#include <sequential-floyd-warshall/graph/edge.hpp>

#include <vector>

struct node
{
   u32 index;
   std::vector<edge> edges;
};

#endif // SEQUENTIAL_FLOYD_WARSHALL_GRAPH_NODE_HPP_
