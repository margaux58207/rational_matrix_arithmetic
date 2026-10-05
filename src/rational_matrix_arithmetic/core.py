"""Exact rational matrix arithmetic using :class:`fractions.Fraction`.

All entries are stored as :class:`~fractions.Fraction`, so every operation
(addition, multiplication, determinant, inverse) produces exact results with
no floating-point rounding. The price is speed: fractions are far slower than
hardware floats, and determinants use plain Laplace expansion which is
``O(n!)``. This is fine for the small matrices (typically up to about 8x8) the
library is designed for.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Union

# Accept either ints (which Fraction normalises automatically) or pre-made
# Fractions in user-supplied data. Strings are deliberately rejected so a
# typo like "1/2" does not silently become 0 under integer interpretation.
_Scalar = Union[int, Fraction]


class Matrix:
    """An immutable matrix of exact rational numbers.

    Construction copies the input data, so later mutation of the source list
    cannot corrupt a matrix. A matrix is rectangular by construction; a
    ragged nested list raises :class:`ValueError` immediately.
    """

    __slots__ = ("_rows", "_nrows", "_ncols")

    def __init__(self, data: Sequence[Sequence[_Scalar]]):
        if not isinstance(data, Sequence):
            raise TypeError("Matrix data must be a sequence of rows")
        if len(data) == 0:
            raise ValueError("Matrix must have at least one row")
        if not isinstance(data[0], Sequence):
            raise TypeError("Each row must be a sequence")
        ncols = len(data[0])
        if ncols == 0:
            raise ValueError("Matrix must have at least one column")

        # Convert up-front so every entry is a Fraction and every check is
        # against the same concrete type.
        rows: List[List[Fraction]] = []
        for row in data:
            if not isinstance(row, Sequence):
                raise TypeError("Each row must be a sequence")
            if len(row) != ncols:
                raise ValueError(
                    f"Matrix is not rectangular: expected {ncols} columns, "
                    f"got {len(row)}"
                )
            converted_row: List[Fraction] = []
            for value in row:
                if isinstance(value, str):
                    # Reject strings explicitly: Fraction(str) accepts them,
                    # but accepting "1/2" here would hide data-entry bugs at
                    # call sites that meant to pass ints.
                    raise TypeError("Entries must be int or Fraction, not str")
                converted_row.append(Fraction(value))
            rows.append(converted_row)

        # Tuples would prevent the mutable-row aliasing the test suite relies
        # on for equality checks after construction; lists let us index
        # consistently without tuple/list branches later.
        self._rows = rows
        self._nrows = len(rows)
        self._ncols = ncols

    @property
    def nrows(self) -> int:
        return self._nrows

    @property
    def ncols(self) -> int:
        return self._ncols

    def entry(self, i: int, j: int) -> Fraction:
        """Return the ``(i, j)`` entry as an exact :class:`Fraction`.

        Indices are 0-based. Out-of-range raises :class:`IndexError`, matching
        Python's native sequence behaviour rather than inventing a custom
        error type.
        """
        return self._rows[i][j]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self._rows == other._rows

    def __hash__(self) -> int:
        # Lists aren't hashable, so we hash the flat tuple of entries.
        return hash(tuple(self.iter_entries()))

    def __repr__(self) -> str:
        return f"Matrix({self._rows!r})"

    def iter_entries(self):
        """Yield every entry in row-major order."""
        for row in self._rows:
            for value in row:
                yield value

    def __add__(self, other: "Matrix") -> "Matrix":
        if not isinstance(other, Matrix):
            return NotImplemented
        if self._nrows != other._nrows or self._ncols != other._ncols:
            raise ValueError(
                f"Cannot add {self._nrows}x{self._ncols} and "
                f"{other._nrows}x{other._ncols} matrices"
            )
        summed = [
            [a + b for a, b in zip(self_row, other_row)]
            for self_row, other_row in zip(self._rows, other._rows)
        ]
        return Matrix(summed)

    def __mul__(self, other: "Matrix") -> "Matrix":
        if not isinstance(other, Matrix):
            return NotImplemented
        if self._ncols != other._nrows:
            raise ValueError(
                f"Cannot multiply {self._nrows}x{self._ncols} by "
                f"{other._nrows}x{other._ncols}: inner dimensions differ"
            )
        a, b = self._rows, other._rows
        n, m, p = self._nrows, self._ncols, other._ncols
        # Pre-allocate the result so each cell is written exactly once; avoids
        # building rows by append-and-extend which is harder to read.
        result = [[Fraction(0) for _ in range(p)] for _ in range(n)]
        for i in range(n):
            a_i = a[i]
            r_i = result[i]
            for k in range(m):
                a_ik = a_i[k]
                if a_ik == 0:
                    continue
                b_k = b[k]
                for j in range(p):
                    r_i[j] += a_ik * b_k[j]
        return Matrix(result)

    def determinant(self) -> Fraction:
        """Return the determinant as an exact :class:`Fraction`.

        Laplace expansion along the first row. ``O(n!)`` but simple, exact,
        and fine for the small matrices this library targets.
        """
        if self._nrows != self._ncols:
            raise ValueError(
                f"Determinant is only defined for square matrices; "
                f"got {self._nrows}x{self._ncols}"
            )
        return self._det(self._rows)

    @staticmethod
    def _det(m: List[List[Fraction]]) -> Fraction:
        n = len(m)
        if n == 1:
            return m[0][0]
        if n == 2:
            # The 2x2 fast path avoids building submatrices that would
            # immediately collapse to scalars.
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        total = Fraction(0)
        first_row = m[0]
        sign = 1
        for j in range(n):
            coeff = first_row[j]
            if coeff == 0:
                # Skipping zero coefficients is not just an optimisation: it
                # avoids constructing a submatrix that cannot contribute.
                sign = -sign
                continue
            sub = [row[:j] + row[j + 1:] for row in m[1:]]
            total += sign * coeff * Matrix._det(sub)
            sign = -sign
        return total

    def inverse(self) -> "Matrix":
        """Return the inverse as an exact rational :class:`Matrix`.

        Uses the adjugate / cofactor method so every cell stays a
        :class:`Fraction`. Raises :class:`ValueError` for non-square
        matrices and for singular matrices (determinant exactly zero).
        """
        if self._nrows != self._ncols:
            raise ValueError(
                f"Inverse is only defined for square matrices; "
                f"got {self._nrows}x{self._ncols}"
            )
        n = self._nrows
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular; no inverse exists")

        if n == 1:
            return Matrix([[Fraction(1) / self._rows[0][0]]])

        # Cofactor matrix C[i][j] = (-1)^(i+j) * M_ij, where M_ij is the
        # determinant of the submatrix with row i and column j removed.
        cofactors: List[List[Fraction]] = [[Fraction(0)] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                sub = [
                    [self._rows[r][c] for c in range(n) if c != j]
                    for r in range(n) if r != i
                ]
                minor = Matrix._det(sub)
                cofactors[i][j] = minor if (i + j) % 2 == 0 else -minor

        # adjugate = transpose of cofactor matrix; inverse = adjugate / det.
        adjugate = [[cofactors[j][i] for j in range(n)] for i in range(n)]
        inverse_rows = [[cell / det for cell in row] for row in adjugate]
        return Matrix(inverse_rows)
