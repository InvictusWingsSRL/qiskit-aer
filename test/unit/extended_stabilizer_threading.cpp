// This code is licensed under the Apache License, Version 2.0.
// See LICENSE.txt in the root of this repository.

#include "simulators/extended_stabilizer/ch_runner.hpp"

#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void check_thread_limits() {
  using CHSimulator::parallel_threads;
  const auto unlimited = std::numeric_limits<std::size_t>::max();
  require(parallel_threads(0) == 1, "zero thread budget must remain serial");
  require(parallel_threads(1) == 1, "explicit serial budget was ignored");
#ifdef _OPENMP
  const int saved = omp_get_max_threads();
  omp_set_num_threads(2);
  require(parallel_threads(unlimited) <= 2, "OpenMP thread budget was ignored");
  omp_set_num_threads(std::numeric_limits<int>::max());
  require(parallel_threads(unlimited) <=
              static_cast<std::size_t>(omp_get_num_procs()),
          "thread budget exceeds the available CPUs");
#if _OPENMP >= 200805
  require(parallel_threads(unlimited) <=
              static_cast<std::size_t>(omp_get_thread_limit()),
          "OMP_THREAD_LIMIT was ignored");
#endif
  omp_set_num_threads(saved);
  std::atomic<bool> nested_serial{true};
  const int outer_threads = parallel_threads(2);
#pragma omp parallel num_threads(outer_threads)
  {
    if (omp_in_parallel() && parallel_threads(unlimited) != 1)
      nested_serial = false;
  }
  require(nested_serial, "nested state updates created another team");
#else
  require(parallel_threads(unlimited) == 1, "serial build requested workers");
#endif

  CHSimulator::Runner small(4);
  small.initialize_omp(unlimited, 100);
  small.initialize_decomposition(5, 1);
  require(small.get_parallel_threads() == 1,
          "decomposition threshold was ignored");
  CHSimulator::Runner non_clifford(4);
  non_clifford.initialize_omp(unlimited, 100);
  non_clifford.initialize_decomposition(2929, 2500);
  require(non_clifford.get_parallel_threads(1024) <= 2,
          "small Metropolis reductions created an oversized team");
  CHSimulator::Runner few_terms(4);
  few_terms.initialize_omp(unlimited, 0);
  few_terms.initialize_decomposition(3, 1);
  require(few_terms.get_parallel_threads() <= 3,
          "more workers than decomposition terms");
}

void check_norm_reduction() {
  std::mt19937_64 rng(918273);
  for (const unsigned qubits : {1u, 4u, 9u}) {
    for (const unsigned terms : {1u, 3u, 131u}) {
      std::vector<CHSimulator::StabilizerState> states;
      std::vector<std::complex<double>> phases;
      for (unsigned term = 0; term < terms; ++term) {
        states.emplace_back(qubits);
        auto &state = states.back();
        for (unsigned q = 0; q < qubits; ++q) {
          if (rng() & 1)
            state.H(q);
          if (rng() & 1)
            state.S(q);
          if (qubits > 1)
            state.CX(q, (q + 1) % qubits);
        }
        phases.push_back(std::polar(1.0 / terms, 0.3 * term));
      }
      // Include a projected zero term, which must not contribute to the norm.
      if (terms > 1) {
        states[0] = CHSimulator::StabilizerState(qubits);
        CHSimulator::pauli_t negative_z;
        negative_z.Z = 1;
        negative_z.e = 2;
        states[0].MeasurePauli(negative_z);
      }
      for (const unsigned samples : {1u, 37u}) {
        std::vector<uint_fast64_t> diag1(samples), diag2(samples);
        std::vector<std::vector<uint_fast64_t>> matrices(
            samples, std::vector<uint_fast64_t>(qubits));
        for (unsigned sample = 0; sample < samples; ++sample) {
          for (unsigned q = 0; q < qubits; ++q) {
            for (unsigned r = q; r < qubits; ++r) {
              if (rng() & 1) {
                matrices[sample][q] |= 1ULL << r;
                matrices[sample][r] |= 1ULL << q;
              }
            }
            diag1[sample] |= matrices[sample][q] & (1ULL << q);
            if (rng() & 1)
              diag2[sample] |= 1ULL << q;
          }
        }
        auto serial = states;
        const double expected =
            CHSimulator::NormEstimate(serial, phases, diag1, diag2, matrices);
        for (const int threads : {1, 2, std::numeric_limits<int>::max()}) {
          auto parallel = states; // Cold transpose caches on every run.
          const double actual = CHSimulator::ParallelNormEstimate(
              parallel, phases, diag1, diag2, matrices, threads);
          require(std::isfinite(actual) &&
                      std::abs(actual - expected) <
                          1e-11 * std::max(1., std::abs(expected)),
                  "parallel norm differs from the serial reference");
        }
      }
    }
  }
}

CHSimulator::Runner sampling_state(std::size_t threads) {
  CHSimulator::Runner runner(2);
  runner.initialize_omp(threads, 100);
  runner.apply_h(0, 0);
  runner.initialize_decomposition(2929, 2500);
  AER::RngEngine branches(6789);
  for (unsigned rank = 0; rank < runner.get_num_states(); ++rank) {
    runner.apply_t(0, branches.rand(), rank);
    runner.apply_h(0, rank);
    runner.apply_h(1, rank);
  }
  return runner;
}

void check_sampling() {
  auto serial = sampling_state(1);
  auto parallel = sampling_state(std::numeric_limits<std::size_t>::max());
  auto repeat = parallel;
  AER::RngEngine serial_rng(8910), parallel_rng(8910), repeat_rng(8910);
  require(parallel.metropolis_estimation(100, 0, parallel_rng).empty(),
          "zero-shot sampling must return an empty result");
  const auto expected = serial.metropolis_estimation(500, 4096, serial_rng);
  const auto actual = parallel.metropolis_estimation(500, 4096, parallel_rng);
  require(actual == repeat.metropolis_estimation(500, 4096, repeat_rng),
          "fixed-seed sampling did not replay");
  require(actual == expected, "bounded sampling changed the seeded samples");
  unsigned zeroes = 0;
  for (const auto sample : actual) {
    require(sample < 4, "sample has an out-of-range bit");
    zeroes += (sample & 1) == 0;
  }
  require(std::abs(zeroes / 4096. - std::pow(std::cos(M_PI / 8), 2)) < 0.06,
          "sampling disagrees with the H-T-H output distribution");
  const auto before = sampling_state(1).statevector();
  const auto after = parallel.statevector();
  for (unsigned i = 0; i < 4; ++i)
    require(std::abs(before[i] - after[i]) < 1e-10,
            "sampling changed the quantum state");

  // Exercise independently initialized chains and their RNG progression too.
  for (unsigned shot = 0; shot < 20; ++shot)
    require(serial.metropolis_estimation(50, serial_rng) ==
                parallel.metropolis_estimation(50, parallel_rng),
            "resampled Metropolis changed the seeded outcomes");
}

} // namespace

int main() {
  try {
    check_thread_limits();
    check_norm_reduction();
    check_sampling();
    std::cout
        << "Extended-stabilizer thread limits, norms and sampling passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
