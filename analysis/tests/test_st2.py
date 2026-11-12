"""Teaching games and an independent exact-rational Möbius oracle for ST2."""
from fractions import Fraction
import importlib.util
from itertools import combinations
import math
from pathlib import Path
import random
import unittest

spec = importlib.util.spec_from_file_location("st2", Path(__file__).resolve().parents[1] / "scripts/st2.py")
st2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st2)


def mobius_oracle(values):
    """Invert the subset lattice with Fractions, then distribute dividends.

    This does not use the implementation's second differences or 1/2,1/6 rule.
    """
    dividends = {}
    for mask in range(16):
        proper = [sub for sub in range(mask) if sub & mask == sub]
        dividends[mask] = Fraction(values[format(mask, "04b")]) - sum(dividends[sub] for sub in proper)
    singleton = {part: dividends[1 << (3 - i)] for i, part in enumerate(st2.PARTS)}
    pairs = {}
    for i, j in combinations(range(4), 2):
        required = (1 << (3 - i)) | (1 << (3 - j))
        pairs[f"P{i + 1}:P{j + 1}"] = sum(
            dividends[mask] / math.comb(mask.bit_count(), 2)
            for mask in range(16) if mask & required == required
        )
    return singleton, pairs


def game(baseline=0, weights=(0, 0, 0, 0), dividends=()):
    """Build a known polynomial on named set membership, without integer masks."""
    result = {}
    for bits in st2.ALL_BITS:
        present = {part for part, bit in zip(st2.PARTS, bits) if bit == "1"}
        result[bits] = baseline + sum(w for part, w in zip(st2.PARTS, weights) if part in present)
        result[bits] += sum(coefficient for players, coefficient in dividends if set(players) <= present)
    return result


class ExactST2Tests(unittest.TestCase):
    def assert_terms(self, result, singles, pairs):
        self.assertEqual(set(result["singletons"]), set(st2.PARTS))
        self.assertEqual(len(result["pairs"]), 6)
        for name in st2.PARTS:
            self.assertAlmostEqual(result["singletons"][name], singles.get(name, 0), places=11)
        for name, value in result["pairs"].items():
            self.assertAlmostEqual(value, pairs.get(name, 0), places=11)
        self.assertTrue(result["efficiency_passed"])
        self.assertAlmostEqual(result["efficiency_error"], 0, places=11)

    def test_constant_has_only_baseline(self):
        result = st2.exact_st2(game(baseline=-7.25))
        self.assert_terms(result, {}, {})
        self.assertEqual((result["baseline"], result["full"]), (-7.25, -7.25))

    def test_nonuniform_signed_additive_game_has_no_pairs(self):
        weights = (1.25, -3.75, 5.5, 0.125)
        result = st2.exact_st2(game(baseline=-2.5, weights=weights))
        self.assert_terms(result, dict(zip(st2.PARTS, weights)), {})

    def test_single_pair_effect_goes_to_that_pair(self):
        result = st2.exact_st2(game(dividends=[(("P1", "P4"), -6)]))
        self.assert_terms(result, {}, {"P1:P4": -6})

    def test_three_way_effect_splits_among_its_three_pairs(self):
        # SII would give 6 to each pair; order-2 ST gives 4.
        result = st2.exact_st2(game(dividends=[(("P1", "P2", "P4"), 12)]))
        self.assert_terms(result, {}, {"P1:P2": 4, "P1:P4": 4, "P2:P4": 4})

    def test_four_way_effect_splits_among_all_six_pairs(self):
        # SII would allocate -4 to each pair, violating ST efficiency.
        result = st2.exact_st2(game(dividends=[(st2.PARTS, -12)]))
        self.assert_terms(result, {}, {f"{a}:{b}": -2 for a, b in combinations(st2.PARTS, 2)})

    def test_input_order_does_not_matter_and_p1_is_leftmost(self):
        values = {bits: 11 * int(bits[0]) + 3 * int(bits[1]) * int(bits[3]) for bits in st2.ALL_BITS}
        before = dict(values)
        result = st2.exact_st2(dict(reversed(list(values.items()))))
        self.assert_terms(result, {"P1": 11}, {"P2:P4": 3})
        self.assertEqual(st2.exact_st2(values), result)
        self.assertEqual(values, before)
        self.assertEqual(st2.ALL_BITS, tuple(format(i, "04b") for i in range(16)))

    def test_random_tables_match_exact_mobius_oracle_and_efficiency(self):
        rng = random.Random(5495)
        for iteration in range(100):
            values = {bits: rng.randrange(-1000, 1001) for bits in st2.ALL_BITS}
            with self.subTest(iteration=iteration):
                singles, pairs = mobius_oracle(values)
                result = st2.exact_st2(values)
                self.assert_terms(result, singles, pairs)
                reconstructed = math.fsum((result["baseline"], *result["singletons"].values(), *result["pairs"].values()))
                self.assertAlmostEqual(reconstructed, values["1111"], places=10)

    def test_large_baseline_with_small_binary_exact_effects(self):
        values = game(baseline=2**40, weights=(0.25, -0.5, 0.75, 1.25),
                      dividends=[(("P1", "P3", "P4"), 3), (st2.PARTS, 6)])
        singles, pairs = mobius_oracle(values)
        self.assert_terms(st2.exact_st2(values), singles, pairs)

    def test_requires_exact_bit_string_key_set(self):
        valid = game()
        invalid = [[], list(valid.items()), {int(k): v for k, v in valid.items()}]
        missing = dict(valid);missing.pop("0001")
        extra = dict(valid);extra["other"] = 0
        malformed = dict(valid);malformed["000x"] = malformed.pop("0001")
        for values in (*invalid, missing, extra, malformed):
            with self.subTest(values=repr(values)[:60]):
                with self.assertRaises((TypeError, ValueError)):
                    st2.exact_st2(values)

    def test_rejects_bool_nonreal_and_nonfinite_scores(self):
        for score in (True, False, None, "1.0", 1 + 0j, math.nan, math.inf, -math.inf):
            values = game();values["1010"] = score
            with self.subTest(score=score):
                with self.assertRaises((TypeError, ValueError)):
                    st2.exact_st2(values)

    def test_rejects_invalid_tolerances(self):
        for field in ("abs_tol", "rel_tol"):
            for tolerance in (-1, math.nan, math.inf, -math.inf, True, None, "0.1"):
                with self.subTest(field=field, tolerance=tolerance):
                    with self.assertRaises((TypeError, ValueError)):
                        st2.exact_st2(game(), **{field: tolerance})
        self.assertTrue(st2.exact_st2(game(), abs_tol=0, rel_tol=0)["efficiency_passed"])

    def test_unrepresentable_arithmetic_fails_without_infinite_results(self):
        values = game();values["0000"] = -1e308;values["1111"] = 1e308
        with self.assertRaisesRegex(ValueError, "finite float range"):
            st2.exact_st2(values)


if __name__ == "__main__":
    unittest.main()
