#ifndef SEQUENTIAL_FLOYD_WARSHALL_GRAPH_EDGE_HPP_
#define SEQUENTIAL_FLOYD_WARSHALL_GRAPH_EDGE_HPP_

#include <parallel-floyd-warshall/types.hpp>

struct edge
{
   i32 weight;
   u32 end;
};

#endif // SEQUENTIAL_FLOYD_WARSHALL_GRAPH_EDGE_HPP_
