# Rational Matrix Arithmetic

A small Python library for matrix addition, multiplication, determinant, and inverse using exact `Fraction` entries, so results never accumulate floating-point error.

## Usage

```python
from rational_matrix_arithmetic import Matrix

a = Matrix([[1, 2], [3, 4]])
b = Matrix([[5, 6], [7, 8]])

a + b            # Matrix([[6, 8], [10, 12]])
a * b            # Matrix([[19, 22], [43, 50]])
a.determinant()  # Fraction(-2, 1)
a.inverse()      # Matrix([[-2, 1], [3/2, -1/2]])
```

## Why

Floating-point matrix arithmetic silently introduces rounding error that compounds across operations. For teaching, verification, or small exact-algebra work, that error is unacceptable. This library stores every entry as a `fractions.Fraction`, so each operation is exact.

The trade-off is speed: fractions are far slower than hardware floats, and determinants use plain Laplace expansion which is `O(n!)`. The library is intended for small matrices, roughly up to 8x8.

## Edge cases

Construction rejects ragged row lengths, empty matrices, and string entries (so a typo like `"1/2"` raises rather than silently becoming 0). Determinant and inverse raise `ValueError` on non-square matrices; inverse raises `ValueError` on singular matrices. Entry indexing is 0-based and out-of-range raises `IndexError`.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

