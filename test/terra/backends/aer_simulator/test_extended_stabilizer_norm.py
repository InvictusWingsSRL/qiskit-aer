# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

"""Regression tests for extended-stabilizer norm estimation."""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, Statevector, pauli_basis, random_clifford
from test.terra.backends.simulator_test_case import SimulatorTestCase


class TestExtendedStabilizerNorm(SimulatorTestCase):
    """Check exact single-term norms and the serial estimation path."""

    def norm_backend(self, threads, **options):
        """Keep circuit and shot execution serial to exercise state threading."""
        return self.backend(
            method="extended_stabilizer",
            max_parallel_threads=threads,
            max_parallel_experiments=1,
            max_parallel_shots=1,
            **options,
        )

    def test_clifford_saves_are_exact_with_one_sample(self):
        """Clifford expectations and probabilities do not need norm samples."""
        for threads in (1, 4):
            backend = self.norm_backend(
                threads,
                extended_stabilizer_norm_estimation_samples=1,
                extended_stabilizer_norm_estimation_repetitions=1,
            )
            for seed in (7, 5832):
                with self.subTest(threads=threads, seed=seed):
                    circuit = random_clifford(4, seed=seed).to_circuit()
                    target = Statevector(circuit)
                    paulis = pauli_basis(4)
                    expected = []
                    for index, pauli in enumerate(paulis):
                        value = target.expectation_value(pauli).real
                        expected.append([value, 1.0 - value**2])
                        circuit.save_expectation_value_variance(
                            pauli, range(4), label=f"p{index}"
                        )
                    circuit.save_amplitudes_squared(range(16), label="probabilities")

                    result = backend.run(
                        transpile(circuit, backend, optimization_level=0), shots=1
                    ).result()
                    self.assertSuccess(result)
                    data = result.data(0)
                    actual = [data[f"p{index}"] for index in range(len(paulis))]
                    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(
                        data["probabilities"], target.probabilities(), atol=1e-12, rtol=0
                    )

    def test_norm_after_projection(self):
        """Saves normalize a projected stabilizer state using its CH scalar."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.reset(0)
        for pauli in ("II", "IZ", "XI"):
            circuit.save_expectation_value_variance(Pauli(pauli), [0, 1], label=pauli)
        circuit.save_amplitudes_squared(range(4), label="probabilities")

        for threads in (1, 4):
            with self.subTest(threads=threads):
                backend = self.norm_backend(
                    threads,
                    extended_stabilizer_norm_estimation_samples=1,
                    extended_stabilizer_norm_estimation_repetitions=1,
                )
                result = backend.run(
                    transpile(circuit, backend, optimization_level=0), shots=16
                ).result()
                self.assertSuccess(result)
                data = result.data(0)
                np.testing.assert_allclose(data["II"], [1, 0], atol=1e-12)
                np.testing.assert_allclose(data["IZ"], [1, 0], atol=1e-12)
                np.testing.assert_allclose(data["XI"], [0, 1], atol=1e-12)
                self.assertAlmostEqual(sum(data["probabilities"]), 1.0)
                np.testing.assert_allclose(
                    np.asarray(data["probabilities"])[[1, 3]], [0, 0], atol=1e-12
                )

    def test_norm_estimation_below_parallel_threshold(self):
        """A small decomposition uses the same samples regardless of thread limit."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.t(0)
        circuit.cx(0, 1)
        circuit.h(1)
        circuit.save_amplitudes_squared(range(4), label="probabilities")

        probabilities = []
        for threads in (1, 4):
            backend = self.norm_backend(
                threads,
                extended_stabilizer_parallel_threshold=100,
                extended_stabilizer_approximation_error=0.5,
                extended_stabilizer_norm_estimation_samples=100,
                extended_stabilizer_norm_estimation_repetitions=3,
            )
            result = backend.run(
                transpile(circuit, backend, optimization_level=0), shots=1
            ).result()
            self.assertSuccess(result)
            probabilities.append(result.data(0)["probabilities"])

        np.testing.assert_array_equal(probabilities[0], probabilities[1])
