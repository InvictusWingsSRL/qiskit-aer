# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimulator Integration Tests
"""
import numpy as np
from ddt import ddt
from test.terra.reference import ref_non_clifford
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, Statevector
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

SUPPORTED_METHODS = [
    "automatic",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "extended_stabilizer",
    "tensor_network",
]


@ddt
class TestNonCliffords(SimulatorTestCase):
    """AerSimulator T and CCX gate tests"""

    # ---------------------------------------------------------------------
    # Test t-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_t_gate_deterministic_default_basis_gates(self, method, device):
        """Test t-gate circuits compiling to backend default basis_gates."""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_non_clifford.t_gate_circuits_deterministic(final_measure=True)
        circuits = transpile(circuits, backend, optimization_level=0)
        result = backend.run(circuits, shots=shots).result()
        targets = ref_non_clifford.t_gate_counts_deterministic(shots)
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_t_gate_nondeterministic_default_basis_gates(self, method, device):
        """Test t-gate circuits compiling to backend default basis_gates."""
        backend = self.backend(
            method=method, device=device, extended_stabilizer_metropolis_mixing_time=50
        )
        shots = 500
        circuits = ref_non_clifford.t_gate_circuits_nondeterministic(final_measure=True)
        circuits = transpile(circuits, backend, optimization_level=0)
        result = backend.run(circuits, shots=shots).result()
        targets = ref_non_clifford.t_gate_counts_nondeterministic(shots)
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    # ---------------------------------------------------------------------
    # Test tdg-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_tdg_gate_deterministic_default_basis_gates(self, method, device):
        """Test tdg-gate circuits compiling to backend default basis_gates."""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_non_clifford.tdg_gate_circuits_deterministic(final_measure=True)
        circuits = transpile(circuits, backend, optimization_level=0)
        result = backend.run(circuits, shots=shots).result()
        targets = ref_non_clifford.tdg_gate_counts_deterministic(shots)
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_tdg_gate_nondeterministic_default_basis_gates(self, method, device):
        """Test tdg-gate circuits compiling to backend default basis_gates."""
        backend = self.backend(
            method=method, device=device, extended_stabilizer_metropolis_mixing_time=50
        )
        shots = 500
        circuits = ref_non_clifford.tdg_gate_circuits_nondeterministic(final_measure=True)
        circuits = transpile(circuits, backend, optimization_level=0)
        result = backend.run(circuits, shots=shots).result()
        targets = ref_non_clifford.tdg_gate_counts_nondeterministic(shots)
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    # ---------------------------------------------------------------------
    # Test ccx-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_ccx_gate_deterministic_default_basis_gates(self, method, device):
        """Test ccx-gate circuits compiling to backend default basis_gates."""
        backend = self.backend(
            method=method, device=device, extended_stabilizer_metropolis_mixing_time=100
        )
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(final_measure=True)
        circuits = transpile(circuits, backend, optimization_level=0)
        result = backend.run(circuits, shots=shots).result()
        targets = ref_non_clifford.ccx_gate_counts_deterministic(shots)
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_ccx_gate_nondeterministic_default_basis_gates(self, method, device):
        """Test ccx-gate circuits compiling to backend default basis_gates."""
        backend = self.backend(
            method=method, device=device, extended_stabilizer_metropolis_mixing_time=100
        )
        shots = 500
        circuits = ref_non_clifford.ccx_gate_circuits_nondeterministic(final_measure=True)
        circuits = transpile(circuits, backend, optimization_level=0)
        result = backend.run(circuits, shots=shots).result()
        targets = ref_non_clifford.ccx_gate_counts_nondeterministic(shots)
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.10 * shots)

    # ---------------------------------------------------------------------
    # Extended-stabilizer non-Clifford regressions
    # ---------------------------------------------------------------------
    def _extended_stabilizer_backend(self, **options):
        defaults = {
            "method": "extended_stabilizer",
            "device": "CPU",
            "seed_simulator": 12345,
            "extended_stabilizer_approximation_error": 0.02,
            "extended_stabilizer_parallel_threshold": 1,
            "max_parallel_threads": 4,
        }
        defaults.update(options)
        return self.backend(**defaults)

    def test_extended_stabilizer_general_z_rotations(self):
        """General P and RZ angles use the sampled Clifford decomposition."""
        backend = self._extended_stabilizer_backend()

        for gate in ("p", "rz"):
            for angle in (np.pi / 3, -np.pi / 3, 2 * np.pi / 3, -2 * np.pi / 3):
                with self.subTest(gate=gate, angle=angle):
                    circuit = QuantumCircuit(1)
                    circuit.h(0)
                    getattr(circuit, gate)(angle, 0)
                    target = Statevector(circuit).data
                    circuit.save_statevector(label="state")

                    result = backend.run(
                        transpile(circuit, backend, optimization_level=0), shots=1
                    ).result()
                    self.assertSuccess(result)
                    self.assertTrue(
                        np.allclose(result.data(0)["state"].data, target, atol=0.08)
                    )

    def test_extended_stabilizer_clifford_rz_global_phase(self):
        """A Clifford-valued RZ retains its gate-level global phase."""
        backend = self._extended_stabilizer_backend()
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.rz(np.pi / 2, 0)
        target = Statevector(circuit).data
        circuit.save_statevector(label="state")

        result = backend.run(
            transpile(circuit, backend, optimization_level=0), shots=1
        ).result()
        self.assertSuccess(result)
        self.assertTrue(np.allclose(result.data(0)["state"].data, target))

    def test_extended_stabilizer_phase_after_decomposition(self):
        """Gate-level phases are applied once, not once per stabilizer term."""
        backend = self._extended_stabilizer_backend()
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.t(0)
        circuit.sx(0)
        target = Statevector(circuit).data
        circuit.save_statevector(label="state")

        result = backend.run(
            transpile(circuit, backend, optimization_level=0), shots=1
        ).result()
        self.assertSuccess(result)
        self.assertTrue(
            np.allclose(result.data(0)["state"].data, target, atol=0.08)
        )

    def test_extended_stabilizer_nonclifford_save_instructions(self):
        """Non-Clifford saves are exposed and normalized consistently."""
        backend = self._extended_stabilizer_backend(
            extended_stabilizer_norm_estimation_samples=500,
            extended_stabilizer_norm_estimation_repetitions=5,
        )
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.t(0)
        circuit.h(0)
        target = Statevector(circuit)
        target_z = target.expectation_value(Pauli("Z")).real
        circuit.save_expectation_value(Pauli("I"), [0], label="identity")
        circuit.save_expectation_value_variance(Pauli("Z"), [0], label="z")
        circuit.save_amplitudes_squared([0, 1], label="probabilities")

        result = backend.run(
            transpile(circuit, backend, optimization_level=0), shots=1
        ).result()
        self.assertSuccess(result)
        data = result.data(0)
        self.assertAlmostEqual(data["identity"], 1.0, delta=0.08)
        self.assertTrue(
            np.allclose(
                data["z"], [target_z, 1.0 - target_z**2], atol=0.12
            )
        )
        self.assertTrue(
            np.allclose(data["probabilities"], target.probabilities(), atol=0.12)
        )

    def test_extended_stabilizer_checkpoints_after_decomposition(self):
        """Clifford and non-Clifford evolution continues after state saves."""
        backend = self._extended_stabilizer_backend()
        circuit = QuantumCircuit(1)
        target_circuit = QuantumCircuit(1)

        circuit.h(0)
        circuit.t(0)
        target_circuit.h(0)
        target_circuit.t(0)
        first_target = Statevector(target_circuit).data
        circuit.save_statevector(label="after_t")

        circuit.h(0)
        target_circuit.h(0)
        second_target = Statevector(target_circuit).data
        circuit.save_statevector(label="after_h")

        circuit.p(-np.pi / 5, 0)
        target_circuit.p(-np.pi / 5, 0)
        third_target = Statevector(target_circuit).data
        circuit.save_statevector(label="after_p")

        result = backend.run(
            transpile(circuit, backend, optimization_level=0), shots=1
        ).result()
        self.assertSuccess(result)
        data = result.data(0)
        self.assertTrue(np.allclose(data["after_t"].data, first_target, atol=0.08))
        self.assertTrue(np.allclose(data["after_h"].data, second_target, atol=0.08))
        self.assertTrue(np.allclose(data["after_p"].data, third_target, atol=0.10))
