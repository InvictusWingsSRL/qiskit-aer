Native regression tests can be built independently of the Python extension:

```bash
cmake -S test/unit -B build-native -DCMAKE_BUILD_TYPE=Release \
  -DAER_JSON_INCLUDE_DIR=/path/to/nlohmann-json/include
cmake --build build-native
ctest --test-dir build-native --output-on-failure
```

They also participate in a normal Aer build configured with `BUILD_TESTS=ON`.
The standalone build accepts `AER_TEST_OPENMP=OFF` to exercise the serial
fallback. GCC and Clang discard unrelated, unused Clifford functions from
transitive headers in that mode; those functions otherwise have existing
OpenMP linker dependencies even when `_OPENMP` is undefined.

The tests cover CPU/runtime thread caps, explicit serial execution, nested
regions, small decompositions, cold transpose caches, parallel versus serial
norm estimates, projected zero terms, seeded sampling replay and quantum-state
preservation. CTest additionally exercises `OMP_THREAD_LIMIT` and dynamic teams.
Assertions remain active in Release builds. No wall-clock thresholds are used.
