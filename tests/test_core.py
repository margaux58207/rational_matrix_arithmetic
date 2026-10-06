import unittest
from fractions import Fraction

from rational_matrix_arithmetic import Matrix


class TestConstruction(unittest.TestCase):
    def test_int_entries_become_fractions(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.entry(0, 0), Fraction(1))
        self.assertIsInstance(m.entry(0, 0), Fraction)

    def test_fraction_entries_preserved(self):
        m = Matrix([[Fraction(1, 2), Fraction(3, 4)], [5, 6]])
        self.assertEqual(m.entry(0, 0), Fraction(1, 2))
        self.assertEqual(m.entry(0, 1), Fraction(3, 4))

    def test_shape_properties(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m.nrows, 2)
        self.assertEqual(m.ncols, 3)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            Matrix([])
        with self.assertRaises(ValueError):
            Matrix([[1, 2], []])

    def test_ragged_raises(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3]])
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3, 4, 5]])

    def test_string_entries_rejected(self):
        with self.assertRaises(TypeError):
            Matrix([["1/2", 2], [3, 4]])


class TestEquality(unittest.TestCase):
    def test_equal_matrices(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[1, 2], [3, 4]])
        self.assertEqual(a, b)

    def test_unequal_matrices(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[1, 2], [3, 5]])
        self.assertNotEqual(a, b)

    def test_different_shapes_not_equal(self):
        a = Matrix([[1, 2]])
        b = Matrix([[1], [2]])
        self.assertNotEqual(a, b)

    def test_not_equal_to_non_matrix(self):
        a = Matrix([[1, 2], [3, 4]])
        self.assertNotEqual(a, "not a matrix")


class TestAddition(unittest.TestCase):
    def test_basic_addition(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        result = a + b
        expected = Matrix([[6, 8], [10, 12]])
        self.assertEqual(result, expected)

    def test_addition_with_fractions(self):
        a = Matrix([[Fraction(1, 2), 0], [0, Fraction(3, 4)]])
        b = Matrix([[Fraction(1, 2), 0], [0, Fraction(1, 4)]])
        result = a + b
        self.assertEqual(result, Matrix([[1, 0], [0, 1]]))

    def test_dimension_mismatch_raises(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            a + b


class TestMultiplication(unittest.TestCase):
    def test_identity_multiplication(self):
        a = Matrix([[1, 2], [3, 4]])
        ident = Matrix([[1, 0], [0, 1]])
        self.assertEqual(a * ident, a)

    def test_2x2_multiplication(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        self.assertEqual(a * b, Matrix([[19, 22], [43, 50]]))

    def test_non_square_multiplication(self):
        a = Matrix([[1, 2, 3], [4, 5, 6]])  # 2x3
        b = Matrix([[1, 0], [0, 1], [1, 1]])  # 3x2
        result = a * b
        expected = Matrix([[4, 5], [10, 11]])
        self.assertEqual(result, expected)

    def test_dimension_mismatch_raises(self):
        a = Matrix([[1, 2, 3]])
        b = Matrix([[1, 2]])
        with self.assertRaises(ValueError):
            a * b


class TestDeterminant(unittest.TestCase):
    def test_1x1_determinant(self):
        self.assertEqual(Matrix([[7]]).determinant(), Fraction(7))

    def test_2x2_determinant(self):
        a = Matrix([[1, 2], [3, 4]])
        self.assertEqual(a.determinant(), Fraction(-2))

    def test_3x3_determinant(self):
        a = Matrix([[2, 0, 0], [0, 3, 0], [0, 0, 4]])
        self.assertEqual(a.determinant(), Fraction(24))

    def test_determinant_with_fractions(self):
        a = Matrix([[Fraction(1, 2), 1], [1, 2]])
        self.assertEqual(a.determinant(), Fraction(0))

    def test_singular_3x3_determinant(self):
        # Rows 1 and 3 are identical -> singular.
        a = Matrix([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
        self.assertEqual(a.determinant(), Fraction(0))

    def test_non_square_determinant_raises(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3, 4], [5, 6]]).determinant()


class TestInverse(unittest.TestCase):
    def test_inverse_of_2x2(self):
        a = Matrix([[4, 7], [2, 6]])
        inv = a.inverse()
        product = a * inv
        self.assertEqual(product, Matrix([[1, 0], [0, 1]]))

    def test_inverse_with_fractions(self):
        a = Matrix([[Fraction(1, 2), Fraction(1, 3)], [0, 1]])
        inv = a.inverse()
        product = a * inv
        self.assertEqual(product, Matrix([[1, 0], [0, 1]]))

    def test_singular_matrix_raises(self):
        a = Matrix([[1, 2], [2, 4]])
        with self.assertRaises(ValueError):
            a.inverse()

    def test_non_square_inverse_raises(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2, 3]]).inverse()


class TestHash(unittest.TestCase):
    def test_equal_matrices_hash_equal(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[1, 2], [3, 4]])
        self.assertEqual(hash(a), hash(b))


if __name__ == "__main__":
    unittest.main()
