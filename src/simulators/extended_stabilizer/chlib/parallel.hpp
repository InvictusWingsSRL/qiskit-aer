/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2026.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 */

#ifndef CHSIMULATOR_PARALLEL_HPP
#define CHSIMULATOR_PARALLEL_HPP

#include <algorithm>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace CHSimulator {

// Respect the caller's budget, the OpenMP runtime and available CPUs. State
// updates inside a parallel circuit/shot executor must not create nested teams.
inline std::size_t parallel_threads(std::size_t requested) {
#ifdef _OPENMP
  if (omp_in_parallel())
    return 1;
  std::size_t limit = std::max<std::size_t>(
      1, std::min({requested, static_cast<std::size_t>(omp_get_max_threads()),
                   static_cast<std::size_t>(omp_get_num_procs())}));
#if _OPENMP >= 200805
  limit = std::min(limit, static_cast<std::size_t>(omp_get_thread_limit()));
#endif
  return limit;
#else
  (void)requested;
  return 1;
#endif
}

} // namespace CHSimulator

#endif
