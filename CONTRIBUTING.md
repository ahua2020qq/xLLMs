# Contributing to nxtLLM

Thank you for your interest in contributing to nxtLLM! This document outlines
the conventions and workflow to follow.

## Code Style: C11

nxtLLM is written in **C11** (ISO/IEC 9899:2011). All contributions must
conform to the following style rules:

- **Indentation**: 4 spaces (no tabs).
- **Line length**: 100 characters maximum.
- **Braces**: K&R style — opening brace on the same line as the control
  statement, closing brace on its own line.
- **Naming**:
  - Functions: `snake_case` (e.g., `page_alloc`, `lru_k_evict`).
  - Types / structs: `snake_case` with `_t` suffix (e.g., `page_t`, `lru_k_t`).
  - Macros / constants: `UPPER_SNAKE_CASE` (e.g., `MAX_PAGES`, `LRU_K_DEFAULT`).
- **Comments**: `/* ... */` for block comments, `//` allowed for single-line
  remarks. Prefer self-documenting code over comments.
- **Includes**: Standard library headers first, then third-party, then project
  headers. Each group separated by a blank line.
- **Error handling**: Functions return `int` (0 on success, negative errno on
  failure). Avoid `assert()` for runtime errors.

### clang-format

A `.clang-format` file is provided in the repository root. Run it before
submitting:

```sh
clang-format -i src/**/*.c include/**/*.h
```

CI will reject PRs that do not pass the format check.

## Development Workflow

### Prerequisites

- **Compiler**: GCC 11+ or Clang 14+ with C11 support.
- **Build system**: CMake 3.20+.
- **Testing framework**: Check (unit-test framework for C).

### Build & Test

```sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
ctest --output-on-failure
```

### Running Individual Tests

```sh
cd build
ctest -R lru_k_test --output-on-failure
```

### Memory Sanitizers

```sh
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON -DENABLE_UBSAN=ON
make -j$(nproc)
ctest
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `master`.
2. **Keep changes focused** — one logical change per PR.
3. **Write tests** for new functionality. Ensure all tests pass.
4. **Run clang-format** and ensure a clean diff.
5. **Update documentation** if your change affects the public API or user-facing
   behavior.
6. **Submit a PR** against the `master` branch with a clear title and
   description.

### PR Checklist

- [ ] Branch is based on an up-to-date `master`.
- [ ] Code compiles with `-Wall -Wextra -Wpedantic` and zero warnings.
- [ ] `clang-format` has been applied.
- [ ] All existing tests pass (`ctest`).
- [ ] New tests cover the change.
- [ ] No sanitizer errors (ASan / UBSan clean).
- [ ] Relevant documentation is updated.
- [ ] Commit messages are clear and follow
      [Conventional Commits](https://www.conventionalcommits.org/).

## Issue Reporting

- **Bugs**: Use the Bug Report template. Include steps to reproduce, expected
  and actual behavior, and environment details.
- **Features**: Use the Feature Request template. Describe the motivation and
  proposed solution.
- **Security**: See [SECURITY.md](SECURITY.md). Do not file public issues for
  security vulnerabilities.

## Code of Conduct

All participants must follow our [Code of Conduct](CODE_OF_CONDUCT.md). Please
report unacceptable behavior to the project maintainers.

## License

By contributing, you agree that your contributions will be licensed under the
same license as the project.
