import numpy as np
from io import StringIO

from hypnotoad.cases import tokamak
from hypnotoad.core.mesh import BoutMesh
from hypnotoad.geqdsk import _geqdsk


def test_tokamak_interpolations():
    """Test interpolations and derivatives"""

    # Define 2D (R,Z) grid
    r1d = np.linspace(1.0, 2.0, 129)
    z1d = np.linspace(-1.0, 1.0, 129)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    # A poloidal flux function
    r0 = 1.5
    z0 = 0.0

    def psi_func(R, Z):
        return np.exp(-((R - r0) ** 2 + (Z - z0) ** 2) / 0.3 ** 2)

    def dpsi_dr(R, Z):
        "Derivative of psi in R"
        return -(2 / 0.3 ** 2) * (R - r0) * psi_func(R, Z)

    def dpsi_dz(R, Z):
        "Derivative of psi in Z"
        return -(2 / 0.3 ** 2) * (Z - z0) * psi_func(R, Z)

    psi2d = psi_func(r2d, z2d)

    def fpol_func(psi):
        "Define a simple profile for poloidal current function f = R * Bt"
        return 1.0 - 0.1 * psi ** 2

    def fpolprime_func(psi):
        "Derivative of fpol"
        return -0.2 * psi

    psi1d = np.linspace(0, 1, 65)
    fpol1d = fpol_func(psi1d)

    eq = tokamak.TokamakEquilibrium(r1d, z1d, psi2d, psi1d, fpol1d, make_regions=False)

    # Check interpolation of psi, f and f' at some locations
    for r, z in [(1.2, 0.1), (1.6, -0.4), (1.8, 0.9)]:
        psi = psi_func(r, z)

        # Poloidal flux
        assert np.isclose(eq.psi(r, z), psi, atol=1e-5)

        # Poloidal current function
        assert np.isclose(eq.fpol(psi), fpol_func(psi))
        assert np.isclose(eq.fpolprime(psi), fpolprime_func(psi))

        # Poloidal magnetic field
        assert np.isclose(eq.Bp_R(r, z), dpsi_dz(r, z) / r, atol=1e-3)
        assert np.isclose(eq.Bp_Z(r, z), -dpsi_dr(r, z) / r, atol=1e-3)

        # vector Grad(psi)/|Grad(psi)|**2
        assert np.isclose(
            eq.f_R(r, z),
            dpsi_dr(r, z) / (dpsi_dr(r, z) ** 2 + dpsi_dz(r, z) ** 2),
            rtol=1e-3,
        )
        assert np.isclose(
            eq.f_Z(r, z),
            dpsi_dz(r, z) / (dpsi_dr(r, z) ** 2 + dpsi_dz(r, z) ** 2),
            rtol=1e-3,
        )


def test_read_geqdsk(tmpdir):
    # Number of mesh points
    nx = 65
    ny = 65

    # Limits of the domain
    Rmin = 1.0
    Rmax = 2.0
    Zmin = -1.0
    Zmax = 1.0

    # Centre of "plasma"
    r0 = 1.1
    z0 = 0.2

    # A poloidal flux function
    def psi_func(R, Z):
        return -1.5 * np.exp(-((R - r0) ** 2 + (Z - z0) ** 2) / 0.3 ** 2)

    def dpsi_dr(R, Z):
        "Derivative of psi in R"
        return -(2 / 0.3 ** 2) * (R - r0) * psi_func(R, Z)

    def dpsi_dz(R, Z):
        "Derivative of psi in Z"
        return -(2 / 0.3 ** 2) * (Z - z0) * psi_func(R, Z)

    psi_boundary = psi_func(1.5, z0)

    def fpol_func(psi):
        "Define a simple profile for poloidal current function f = R * Bt"
        psi = np.clip(psi, None, psi_boundary)
        return 1.0 - 0.1 * psi ** 2

    def fpolprime_func(psi):
        "Derivative of fpol"
        return np.where(psi < psi_boundary, -0.2 * psi, 0.0)

    psi1d = np.linspace(psi_func(r0, z0), psi_boundary, nx)

    data = {
        "nx": nx,
        "ny": ny,
        "rdim": Rmax - Rmin,
        "zdim": Zmax - Zmin,
        "rleft": Rmin,
        "rcentr": r0,
        "bcentr": 1.0,
        "zmid": 0.5 * (Zmax + Zmin),
        "rmagx": r0,
        "zmagx": z0,
        "simagx": psi_func(r0, z0),
        "sibdry": psi_func(1.5, z0),
        "cpasma": 1234521,
        "fpol": fpol_func(psi1d),
        "pres": np.zeros(nx),
        "qpsi": np.zeros(nx),
        "psi": psi_func(
            *np.meshgrid(
                np.linspace(Rmin, Rmax, nx), np.linspace(Zmin, Zmax, ny), indexing="ij"
            )
        ),
    }

    # Write to string
    output = StringIO()
    _geqdsk.write(data, output)

    # Move to the beginning of the buffer
    output.seek(0)

    # Read geqdsk from StringIO
    eq = tokamak.read_geqdsk(output)

    # Check interpolation of psi, f and f' at some locations
    for r, z in [(1.2, 0.1), (1.6, -0.4), (1.8, 0.9)]:
        psi = psi_func(r, z)

        # Poloidal flux
        assert np.isclose(eq.psi(r, z), psi, atol=1e-5)

        # Poloidal current function
        assert np.isclose(eq.fpol(psi), fpol_func(psi))
        assert np.isclose(eq.fpolprime(psi), fpolprime_func(psi))

        # Poloidal magnetic field
        assert np.isclose(eq.Bp_R(r, z), dpsi_dz(r, z) / r, atol=1e-3)
        assert np.isclose(eq.Bp_Z(r, z), -dpsi_dr(r, z) / r, atol=1e-3)

        # vector Grad(psi)/|Grad(psi)|**2
        assert np.isclose(
            eq.f_R(r, z),
            dpsi_dr(r, z) / (dpsi_dr(r, z) ** 2 + dpsi_dz(r, z) ** 2),
            rtol=1e-3,
        )
        assert np.isclose(
            eq.f_Z(r, z),
            dpsi_dz(r, z) / (dpsi_dr(r, z) ** 2 + dpsi_dz(r, z) ** 2),
            rtol=1e-3,
        )

    # Create BoutMesh
    mesh = BoutMesh(eq, {})

    # Write to temporary directory
    gridfile_name = tmpdir.join("bout.grd.nc")
    mesh.writeGridFile(gridfile_name)

    # Check geqdsk input was save correctly
    with DataFile(gridfile_name) as f:
        geqdsk_str = f["hypnotoad_geqdsk_file_contents"]
    output.seek(0)
    assert geqdsk_str == output.read()


def test_bounding():
    nx = 65
    ny = 65

    # Limits of the domain
    Rmin = 1.1
    Rmax = 2.32
    Zmin = -1.314
    Zmax = 0.93

    eq = tokamak.TokamakEquilibrium(
        np.linspace(Rmin, Rmax, nx),
        np.linspace(Zmin, Zmax, ny),
        np.zeros((nx, ny)),  # psi2d
        [],
        [],  # psi1d, fpol
        make_regions=False,
    )

    assert np.isclose(eq.Rmin, Rmin)
    assert np.isclose(eq.Rmax, Rmax)
    assert np.isclose(eq.Zmin, Zmin)
    assert np.isclose(eq.Zmax, Zmax)


def test_xpoint():
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, ny)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = -0.3

    # This has two O-points, and one x-point at (r0, z0)
    def psi_func(R, Z):
        return np.exp(-((R - r0) ** 2 + (Z - z0 - 0.3) ** 2) / 0.3 ** 2) + np.exp(
            -((R - r0) ** 2 + (Z - z0 + 0.3) ** 2) / 0.3 ** 2
        )

    eq = tokamak.TokamakEquilibrium(
        r1d, z1d, psi_func(r2d, z2d), [], [], make_regions=False  # psi1d, fpol
    )

    assert len(eq.x_points) == 1
    assert len(eq.psi_sep) == 1

    assert np.isclose(eq.x_points[0].R, r0, atol=1.0 / nx)
    assert np.isclose(eq.x_points[0].Z, z0, atol=1.0 / ny)

    assert np.isclose(eq.psi_sep[0], psi_func(r0, z0), rtol=1.0 / nx)


def test_wall_anticlockwise():
    nx = 65
    ny = 65

    # Limits of the domain
    Rmin = 1.1
    Rmax = 2.32
    Zmin = -1.314
    Zmax = 0.93

    # Wall going anti-clockwise
    wall = [(Rmin, Zmin), (Rmax, Zmin), (Rmax, Zmax), (Rmin, Zmax)]

    eq = tokamak.TokamakEquilibrium(
        np.linspace(Rmin, Rmax, nx),
        np.linspace(Zmin, Zmax, ny),
        np.zeros((nx, ny)),  # psi2d
        [],
        [],  # psi1d, fpol
        wall=wall,
        make_regions=False,
    )
    assert len(eq.wall) == 4
    # Wall ordering unchanged
    assert eq.wall[0].R == Rmin
    assert eq.wall[0].Z == Zmin
    assert eq.wall[1].R == Rmax
    assert eq.wall[1].Z == Zmin


def test_wall_clockwise():
    nx = 65
    ny = 65

    # Limits of the domain
    Rmin = 1.1
    Rmax = 2.32
    Zmin = -1.314
    Zmax = 0.93

    # Wall going clockwise. This should be reversed by TokamakEquilibrium
    wall = [(Rmin, Zmax), (Rmax, Zmax), (Rmax, Zmin), (Rmin, Zmin)]

    eq = tokamak.TokamakEquilibrium(
        np.linspace(Rmin, Rmax, nx),
        np.linspace(Zmin, Zmax, ny),
        np.zeros((nx, ny)),  # psi2d
        [],
        [],  # psi1d, fpol
        wall=wall,
        make_regions=False,
    )

    assert len(eq.wall) == 4
    # Wall ordering reversed
    assert eq.wall[0].R == Rmin
    assert eq.wall[0].Z == Zmin
    assert eq.wall[1].R == Rmax
    assert eq.wall[1].Z == Zmin


###################################################################
# These routines create a TokamakEquilbrium, but do not generate
# the regions. This is to allow earlier stages to be tested.


def make_lower_single_null():
    nx = 65
    ny = 65

    r1d = np.linspace(1.2, 1.8, nx)
    z1d = np.linspace(-0.5, 0.5, ny)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = -0.3

    # This has two O-points, and one x-point at (r0, z0)
    def psi_func(R, Z):
        return np.exp(-((R - r0) ** 2 + (Z - z0 - 0.3) ** 2) / 0.3 ** 2) + np.exp(
            -((R - r0) ** 2 + (Z - z0 + 0.3) ** 2) / 0.3 ** 2
        )

    return tokamak.TokamakEquilibrium(
        r1d, z1d, psi_func(r2d, z2d), [], [], make_regions=False  # psi1d, fpol
    )


def make_upper_single_null():
    nx = 65
    ny = 65

    r1d = np.linspace(1.2, 1.8, nx)
    z1d = np.linspace(-0.5, 0.5, ny)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = -0.3

    # This has two O-points, and one x-point at (r0, z0)
    def psi_func(R, Z):
        Z = -Z  # Upside-down
        return np.exp(-((R - r0) ** 2 + (Z - z0 - 0.3) ** 2) / 0.3 ** 2) + np.exp(
            -((R - r0) ** 2 + (Z - z0 + 0.3) ** 2) / 0.3 ** 2
        )

    return tokamak.TokamakEquilibrium(
        r1d, z1d, psi_func(r2d, z2d), [], [], make_regions=False  # psi1d, fpol
    )


def make_connected_double_null():
    nx = 65
    ny = 65

    r1d = np.linspace(1.2, 1.8, nx)
    z1d = np.linspace(-0.5, 0.5, ny)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.3

    # This has two X-points
    def psi_func(R, Z):
        return (
            np.exp(-((R - r0) ** 2 + Z ** 2) / 0.3 ** 2)
            + np.exp(-((R - r0) ** 2 + (Z + 2 * z0) ** 2) / 0.3 ** 2)
            + np.exp(-((R - r0) ** 2 + (Z - 2 * z0) ** 2) / 0.3 ** 2)
        )

    return tokamak.TokamakEquilibrium(
        r1d, z1d, psi_func(r2d, z2d), [], [], make_regions=False  # psi1d, fpol
    )


def make_lower_double_null():
    nx = 65
    ny = 65

    r1d = np.linspace(1.2, 1.8, nx)
    z1d = np.linspace(-0.5, 0.5, ny)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.3

    def psi_func(R, Z):
        return (
            -np.exp(-((R - r0) ** 2 + Z ** 2) / 0.3 ** 2)
            - np.exp(-((R - r0) ** 2 + (Z + 2 * z0) ** 2) / 0.3 ** 2)
            - np.exp(-((R - r0) ** 2 + (Z - 2 * z0 - 0.003) ** 2) / 0.3 ** 2)
        )

    return tokamak.TokamakEquilibrium(
        r1d, z1d, psi_func(r2d, z2d), [], [], make_regions=False  # psi1d, fpol
    )


def make_upper_double_null():
    nx = 65
    ny = 65

    r1d = np.linspace(1.2, 1.8, nx)
    z1d = np.linspace(-0.5, 0.5, ny)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.3

    def psi_func(R, Z):
        return (
            np.exp(-((R - r0) ** 2 + Z ** 2) / 0.3 ** 2)
            + np.exp(-((R - r0) ** 2 + (Z + 2 * z0 + 0.002) ** 2) / 0.3 ** 2)
            + np.exp(-((R - r0) ** 2 + (Z - 2 * z0) ** 2) / 0.3 ** 2)
        )

    return tokamak.TokamakEquilibrium(
        r1d, z1d, psi_func(r2d, z2d), [], [], make_regions=False  # psi1d, fpol
    )


def make_upper_double_null_largesep(settings={}):
    """UDN with larger separation between X-points.
    With psinorm = 1.1 this should be single null
    With psinorm = 1.2 it's double null"""

    nx = 65
    ny = 65

    r1d = np.linspace(1.2, 1.8, nx)
    z1d = np.linspace(-0.5, 0.5, ny)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = 0.3

    def psi_func(R, Z):
        return (
            np.exp(-((R - r0) ** 2 + Z ** 2) / 0.3 ** 2)
            + np.exp(-((R - r0) ** 2 + (Z + 2 * z0 + 0.02) ** 2) / 0.3 ** 2)
            + np.exp(-((R - r0) ** 2 + (Z - 2 * z0) ** 2) / 0.3 ** 2)
        )

    return tokamak.TokamakEquilibrium(
        r1d, z1d, psi_func(r2d, z2d), [], [], make_regions=False, settings=settings
    )


###################################################################


def test_findlegs():
    eq = make_lower_single_null()
    legs = eq.findLegs(eq.x_points[0])

    # Check both inner and outer legs are present
    assert "inner" in legs
    assert "outer" in legs

    # The first point in both legs should be the same (the X-point)
    assert legs["inner"][0] == legs["outer"][0]

    # The inner leg should terminate at a smaller major radius than the outer leg
    assert legs["inner"][-1].R < legs["outer"][-1].R


def test_findlegs_upper():
    eq = make_upper_single_null()
    legs = eq.findLegs(eq.x_points[0])

    # Check both inner and outer legs are present
    assert "inner" in legs
    assert "outer" in legs

    # The first point in both legs should be the same (the X-point)
    assert legs["inner"][0] == legs["outer"][0]

    # The inner leg should terminate at a smaller major radius than the outer leg
    assert legs["inner"][-1].R < legs["outer"][-1].R


def test_makeregions_lsn():
    eq = make_lower_single_null()
    eq.makeRegions()

    assert len(eq.regions) == 3


def test_makeregions_usn():
    eq = make_upper_single_null()
    eq.makeRegions()

    assert len(eq.regions) == 3


def test_makeregions_cdn():
    eq = make_connected_double_null()
    eq.makeRegions()

    assert len(eq.regions) == 6


def test_makeregions_udn():
    eq = make_upper_double_null()
    eq.makeRegions()

    assert len(eq.regions) == 6


def test_makeregions_ldn():
    eq = make_lower_double_null()
    eq.makeRegions()

    assert len(eq.regions) == 6


def test_makeregions_udn_largesep_1():
    eq = make_upper_double_null_largesep()
    eq.makeRegions()
    assert len(eq.regions) == 3  # Only one X-point in range -> single null


def test_makeregions_udn_largesep_2():
    eq = make_upper_double_null_largesep(settings={"psinorm_sol": 1.2})
    eq.makeRegions()
    assert len(eq.regions) == 6  # Becomes double null
