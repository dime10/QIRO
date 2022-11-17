## Testing

Find tests for the MLIR compiler here, such as tests for the printing and parsing of IR operation, as well as for IR passes and optimizations.

Most tests have not been automated and need but to be run and verified manually, but the two test files `test.mlir` and `testssa.mlir` are automatically run through the *quantum-opt* utility upon every build to ensure that all operations round-trip correctly.
