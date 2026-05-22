#!/usr/bin/env python3
"""Generate reference PSWF values at high precision via mpmath.

The output XML is consumed by src/gromacs/math/tests/pswf_test.cpp.
Run once before building tests; commit the generated XML.
"""

from __future__ import annotations

import mpmath as mp
import xml.etree.ElementTree as ET
from pathlib import Path

mp.mp.dps = 50

# c-values to validate against. Cover paper 3 Table 2 + boundary stress points.
C_VALUES = [mp.mpf("1.0"), mp.sqrt(3), mp.mpf("5.0"), mp.mpf("9.5392"),
            mp.mpf("12.024"), mp.mpf("14.471"), mp.mpf("16.0"), mp.mpf("20.0")]

# Sample x-points within [-1, 1] (PSWF support).
X_POINTS = [
    mp.mpf("0"),
    mp.mpf("0.1"), mp.mpf("0.25"), mp.mpf("0.5"),
    mp.mpf("0.75"), mp.mpf("0.9"), mp.mpf("0.99"),
    mp.mpf("-0.5"),
]


NS_BY_C_BUCKET = [
    48, 64, 80, 92, 106, 120, 130, 144, 156, 168,
    178, 190, 202, 214, 224, 236, 248, 258, 268, 280,
]


def matrix_coefficients(lam: mp.mpf, n: int, c: mp.mpf):
    """FINUFFT/Rokhlin even-Legendre tridiagonal coefficients."""

    dim = n // 2 + 2
    lower = [mp.mpf("0")] * dim
    diag = [mp.mpf("0")] * dim
    upper = [mp.mpf("0")] * dim
    for k in range(dim):
        if 2 * k > n + 2:
            break
        order = mp.mpf(2 * k)
        alpha0 = order * (order - 1) / ((2 * order + 1) * (2 * order - 1))
        beta0 = (
            (order + 1) * (order + 1) / (2 * order + 3)
            + order * order / (2 * order - 1)
        ) / (2 * order + 1)
        gamma0 = (order + 1) * (order + 2) / ((2 * order + 1) * (2 * order + 3))

        lower[k] = -c * c * alpha0
        diag[k] = lam - order * (order + 1) - c * c * beta0
        upper[k] = -c * c * gamma0

        if k != 0:
            lower[k] *= mp.sqrt((2 * k + mp.mpf("0.5")) / (2 * k - mp.mpf("1.5")))
        upper[k] *= mp.sqrt((2 * k + mp.mpf("0.5")) / (2 * k + mp.mpf("2.5")))

    return lower, diag, upper


def leading_eigen_shift(n: int, c: mp.mpf) -> mp.mpf:
    """Return the shifted spectral parameter used by the inverse iteration."""

    dim = n // 2
    lower, diag, _ = matrix_coefficients(mp.mpf("0"), n, c)
    matrix = mp.zeros(dim)
    for i in range(dim):
        matrix[i, i] = diag[i]
        if i + 1 < dim:
            matrix[i, i + 1] = lower[i + 1]
            matrix[i + 1, i] = lower[i + 1]
    eigenvalues, _ = mp.eigsy(matrix)
    return -eigenvalues[dim - 1] + mp.mpf("1e-8")


def factor_tridiagonal(diag, lower, upper, dim: int):
    down = [mp.mpf("0")] * (dim + 2)
    up = [mp.mpf("0")] * (dim + 2)
    inv_diag = [mp.mpf("0")] * (dim + 2)
    diag = list(diag)
    for i in range(dim - 1):
        factor = lower[i + 1] / diag[i]
        diag[i + 1] -= upper[i] * factor
        down[i] = factor
        up[i + 1] = upper[i] / diag[i + 1]
        inv_diag[i + 1] = 1 / diag[i + 1]
    inv_diag[0] = 1 / diag[0]
    return down, up, inv_diag


def solve_factored(down, up, inv_diag, rhs, dim: int):
    rhs = list(rhs)
    for i in range(dim - 1):
        rhs[i + 1] -= down[i] * rhs[i]
    for i in range(dim - 1, 0, -1):
        rhs[i - 1] -= rhs[i] * up[i]
        rhs[i] *= inv_diag[i]
    rhs[0] *= inv_diag[0]
    return rhs


def build_legendre_coefficients(c: mp.mpf):
    bucket = int(c / 10)
    n = NS_BY_C_BUCKET[bucket] if bucket < len(NS_BY_C_BUCKET) else int(c * mp.mpf("1.5"))
    dim = n // 2
    lam = leading_eigen_shift(n, c)
    lower, diag, upper = matrix_coefficients(lam, n, c)
    down, up, inv_diag = factor_tridiagonal(diag, lower, upper, dim)

    coeffs = [mp.mpf("1")] * (dim + 3)
    for _ in range(4):
        coeffs = solve_factored(down, up, inv_diag, coeffs, dim)
        norm = mp.sqrt(mp.fsum(coeffs[j] * coeffs[j] for j in range(dim)))
        for j in range(dim):
            coeffs[j] /= norm

    last = 0
    for i in range(dim):
        if abs(coeffs[i]) > mp.mpf("1e-16"):
            last = i
        coeffs[i] *= mp.sqrt(2 * i + mp.mpf("0.5"))
    return coeffs[:last + 1]


def recurrence_coefficients(size: int):
    coefs = [(mp.mpf("0"), mp.mpf("0"), mp.mpf("0"))] * size
    for i in range(1, size):
        ell = mp.mpf(2 * i - 1)
        coefs[i] = (
            ((2 * ell - 1) * (2 * ell + 1)) / (ell * (ell + 1)),
            ((2 * ell + 1) * (ell - 1) * (ell - 1) + ell * ell * (2 * ell - 3))
            / (ell * (ell + 1) * (2 * ell - 3)),
            ((2 * ell + 1) * (ell - 1) * (ell - 2))
            / (ell * (ell + 1) * (2 * ell - 3)),
        )
    return coefs


def eval_raw(coeffs, recurrences, x: mp.mpf) -> mp.mpf:
    x_sq = x * x
    pjm1 = mp.mpf("0")
    pjm2 = mp.mpf("1")
    value = coeffs[0]
    i = 1
    while i + 1 < len(recurrences):
        pjm1 = pjm2 * (x_sq * recurrences[i][0] - recurrences[i][1]) - pjm1 * recurrences[i][2]
        value += coeffs[i] * pjm1
        pjm2 = (
            pjm1 * (x_sq * recurrences[i + 1][0] - recurrences[i + 1][1])
            - pjm2 * recurrences[i + 1][2]
        )
        value += coeffs[i + 1] * pjm2
        i += 2
    while i < len(recurrences):
        p = pjm2 * (x_sq * recurrences[i][0] - recurrences[i][1]) - pjm1 * recurrences[i][2]
        value += coeffs[i] * p
        pjm1 = pjm2
        pjm2 = p
        i += 1
    return value


def prolate0(c: mp.mpf, x: mp.mpf) -> mp.mpf:
    """Reference psi_0^c(x), normalized so psi_0^c(0)=1."""

    coeffs = build_legendre_coefficients(c)
    recurrences = recurrence_coefficients(len(coeffs))
    return eval_raw(coeffs, recurrences, x) / eval_raw(coeffs, recurrences, mp.mpf("0"))


def main() -> None:
    root = ET.Element("PswfRefdata", precision="50")
    for c in C_VALUES:
        c_node = ET.SubElement(root, "CValue", c=mp.nstr(c, 20))
        for x in X_POINTS:
            sample = ET.SubElement(c_node, "Sample", x=mp.nstr(x, 20))
            sample.text = mp.nstr(prolate0(c, x), 30)

    out = Path("src/gromacs/math/tests/refdata/pswf_reference.xml")
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
