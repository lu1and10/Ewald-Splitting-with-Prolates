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


def prolate0(c: mp.mpf, x: mp.mpf) -> mp.mpf:
    """Reference psi_0^c(x), normalized so psi_0^c(0)=1.

    This builds the even-Legendre tridiagonal matrix for the lowest-order PSWF,
    diagonalizes it at high precision, and reconstructs the eigenfunction.
    """

    n_max = max(44, int(mp.ceil(2 * c)) + 24)
    if n_max % 2:
        n_max += 1
    even_orders = list(range(0, n_max, 2))
    dim = len(even_orders)

    matrix = mp.zeros(dim)
    for row, n_int in enumerate(even_orders):
        n = mp.mpf(n_int)
        matrix[row, row] = (
            n * (n + 1)
            + c**2 * (2 * n * (n + 1) - 1)
            / ((2 * n - 1) * (2 * n + 3))
        )
        if row + 1 < dim:
            off = (
                -c**2 * (n + 1) * (n + 2)
                / ((2 * n + 1) * (2 * n + 3)
                   * mp.sqrt((2 * n + 1) * (2 * n + 5)))
            )
            matrix[row, row + 1] = off
            matrix[row + 1, row] = off

    eigenvalues, eigenvectors = mp.eigsy(matrix)
    lowest = min(range(dim), key=lambda idx: eigenvalues[idx])
    coeffs = [eigenvectors[row, lowest] for row in range(dim)]

    def eval_coeffs(point: mp.mpf) -> mp.mpf:
        value = mp.mpf("0")
        for idx, coeff in enumerate(coeffs):
            value += coeff * mp.legendre(2 * idx, point)
        return value

    value0 = eval_coeffs(mp.mpf("0"))
    if value0 < 0:
        value0 = -value0
        coeffs = [-coeff for coeff in coeffs]

    return eval_coeffs(x) / value0


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
