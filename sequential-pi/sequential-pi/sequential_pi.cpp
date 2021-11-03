#include <cstdint>
#include <iostream>
#include <limits>
#include <random>

#include <mpi.h>

using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f64 = double;

static constexpr u64 sample_count = 10000;
static constexpr f64 circle_center = 0.5;
static constexpr f64 circle_radius = 0.5;
static constexpr f64 desired_pi = 3.141592;
static constexpr f64 convergence_epsilon = 0.000001;

auto is_within_circle(f64 x, f64 y) -> bool
{
   const f64 x_dir = circle_center - x;
   const f64 y_dir = circle_center - y;

   const f64 distance_to_center = std::sqrt(x_dir * x_dir + y_dir * y_dir);

   return distance_to_center <= circle_radius;
}

auto is_converging(f64 value) -> bool
{
   const f64 adjusted_value = value - desired_pi;

   return adjusted_value < convergence_epsilon and adjusted_value > -convergence_epsilon;
}

auto main(int argc, char *argv[]) -> int
{
   int process_id = 0;
   int process_count = 0;

   MPI_Init(&argc, &argv);

   const f64 start_time = MPI_Wtime();

   MPI_Comm_size(MPI_COMM_WORLD, &process_count);
   MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

   std::random_device rd;
   auto random_engine = std::default_random_engine(rd());
   auto distribution = std::uniform_real_distribution<f64>(circle_center - circle_radius,
                                                           circle_center + circle_radius);

   u64 iteration_count = 1;
   u64 circle_hits = 0;
   f64 pi = 0.0;
   do
   {
      for (u64 i = 0; i < sample_count; ++i)
      {
         const f64 x = distribution(random_engine);
         const f64 y = distribution(random_engine);

         if (is_within_circle(x, y))
         {
            ++circle_hits;
         }
      }

      // NOLINTNEXTLINE
      pi = 4.0f * static_cast<f64>(circle_hits) / static_cast<f64>(sample_count * iteration_count);
      ++iteration_count;
   } while (!is_converging(pi));

   const f64 elapsed_time = MPI_Wtime() - start_time;

   MPI_Finalize();

   std::cout << "iteration to converge: " << iteration_count << '\n';
   std::cout << "total samples: " << iteration_count * sample_count << '\n';
   std::cout << "result: " << pi << '\n';
   std::cout << "elapsed time: " << elapsed_time << '\n';

   return EXIT_SUCCESS;
}
