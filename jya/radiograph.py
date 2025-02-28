from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
from jax import numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float

from .math_utils import SphericalVector, image_plane_basis, planck_integral

__all__ = ["Radiograph"]
EMISS_CONST = 4.113203268070765e17  # "W/cm^2/keV^4"


class Radiograph:
    def __init__(
        self,
        dimension: int,
        opacity_tables: dict[
            str | int,
            tuple[
                Float[Array, "n_freq"],
                Float[Array, "n_temp"],
                Float[Array, "n_dens"],
                Float[Array, "n_freq-1 n_temp n_dens"],
                Float[Array, "n_freq n_temp n_dens"],
            ],
        ],
    ):
        """
        Parameters
        ----------

        dimension : int
            The dimension of the input data for the radiograph. Must be 1 or 3.

        opacity_tables : dict[str | int, tuple[Float[Array, "n_freq"], Float[Array, "n_temp"], Float[Array, "n_dens"], Float[Array, "n_freq n_temp n_dens"], Float[Array, "n_freq n_temp n_dens"]]
            A dictionary of opacity tables for different materials. The keys are the material IDs and the values are tuples of arrays (energy_boundaries, temp, dens, ross_opacity, emiss_opacity).
            The units of the arrays are as follows:
            - energy_boundaries: keV
            - temp: keV
            - dens: 1/cm^3
            - ross_opacity: 1/cm
            - emiss_opacity: 1/cm

        """
        self.energy_spacing, self.e_center, self.opacity_interpolator = (
            build_opacity_interpolator(dimension, opacity_tables)
        )
        self.dimension = dimension

    def build_projector(
        self,
        image_axis: Float[Array, "nr"],
        theta_detector: float | None = None,
        phi_detector: float | None = None,
        *,
        time_integrate: bool = True,
        include_opacity: bool = True,
    ) -> Callable[
        ...,
        tuple[Float[Array, ...], Float[Array, ...]],
    ]:
        """
        Builds a projector function that generates a synthetic radiograph from a plasma input

        Parameters
        ----------
        image_axis : Float[Array, "nr"]
            The axis of the synthetic radiograph image in microns.
        theta_detector : float | None
            The polar angle of the detector in radians. Required for 3D radiographs.
        phi_detector : float | None
            The azimuthal angle of the detector in radians. Required for 3D radiographs.
        time_integrate : bool
            If True, the projector will integrate over time. If False, the projector will return a time-resolved radiograph.
        include_opacity : bool
            If True, the projector will include opacity in the calculation. If False, the projector will only include emissivity.

        Returns
        -------
        Callable[..., tuple[Float[Array, "..."], Float[Array, "..."]]]
            The projector function that returns the photon energies and the corresponding radiograph
        """
        if self.dimension == 1:
            return build_projector_1d(
                image_axis,
                self.opacity_interpolator,
                self.e_center,
                time_integrate=time_integrate,
                include_opacity=include_opacity,
            )
        if self.dimension == 3:
            assert theta_detector is not None
            assert phi_detector is not None
            return build_projector_3d(
                image_axis,
                theta_detector,
                phi_detector,
                self.opacity_interpolator,
                self.e_center,
                time_integrate=time_integrate,
                include_opacity=include_opacity,
            )
        raise NotImplementedError


def build_opacity_interpolator(
    dimension: int,
    opacity_tables: dict[
        str | int,
        tuple[
            Float[Array, "n_freq"],
            Float[Array, "n_temp"],
            Float[Array, "n_dens"],
            Float[Array, "n_freq-1 n_temp n_dens"],
            Float[Array, "n_freq n_temp n_dens"],
        ],
    ],
) -> tuple[
    Float[Array, "n_freq - 1"],
    Float[Array, "n_freq - 1"],
    Callable[
        [
            Float[Array, "*elements"],
            Float[Array, "*elements"],
            Float[Array, "*elements"],
        ],
        tuple[Float[Array, "*elements"], Float[Array, "*elements"]],
    ],
]:
    """
    Builds an interpolator for opacity tables

    Parameters
    ----------
    dimension : int
        The dimension of the input data for the radiograph. Must be 1 or 3.
    opacity_tables : dict[str | int, tuple[Float[Array, "n_freq"], Float[Array, "n_temp"], Float[Array, "n_dens"], Float[Array, "n_freq n_temp n_dens"], Float[Array, "n_freq n_temp n_dens"]]
        A dictionary of opacity tables for different materials. The keys are the material IDs and the values are tuples of arrays (energy_boundaries, temp, dens, ross_opacity, emiss_opacity).
        The units of the arrays are as follows:
        - energy_boundaries: keV
        - temp: keV
        - dens: 1/cm^3
        - ross_opacity: 1/cm
        - emiss_opacity: 1/cm

    Returns
    -------
    energy_spacing: Float[Array, "n_freq-1"]
        The energy spacing of the opacity tables
    energy_center: Float[Array, "n_freq-1"]
        The energy center of the opacity tables
    interpolator: Callable[[Float[Array, "*elements"], Float[Array, "*elements"], Float[Array, "*elements"]], tuple[Float[Array, "*elements"], Float[Array, "*elements"]]
        The interpolator function for the opacity tables that takes  density, temperature, and material fractions as input and returns the emission density and transport opacity.

    """
    table_interpolants = {}
    shapes = None
    energies = None
    for material_id, table in opacity_tables.items():
        interpolant, shape, energy = get_table_interpolant(*table)
        if shapes is None and energies is None:
            shapes = shape
            energies = energy
        else:
            assert shapes == shape
            np.testing.assert_allclose(energy, energies)  # type: ignore
        table_interpolants[material_id] = interpolant
    assert energies is not None
    _interpolant = jax.jit(
        lambda idx, *v: jax.lax.switch(idx, list(table_interpolants.values()), *v)
    )
    _energy_spacing = jnp.expand_dims(
        jnp.diff(energies), axis=tuple(range(1, dimension + 1))
    )  # type: ignore

    @jax.jit
    def func(
        dens: Float[Array, "*elements"],
        temp: Float[Array, "*elements"],
        fracs: Float[Array, "n_mats *elements"],
    ) -> tuple[Float[Array, "*elements"], Float[Array, "*elements"]]:
        opac = jax.lax.fori_loop(
            0,
            len(fracs),
            lambda i, total_opacity: total_opacity
            + (_interpolant(i, dens, temp) * fracs[i]),
            jnp.zeros((2, _energy_spacing.shape[0], *(dens.shape))),
        )
        emission_density = opac[1] / _energy_spacing
        transport_opacity = opac[0]
        return emission_density, transport_opacity

    return jnp.diff(energies), 0.5 * (energies[1:] + energies[:-1]), func


def get_table_interpolant(
    energy: Float[Array, "n_freq"],
    temperature: Float[Array, "n_temp"],
    density: Float[Array, "n_dens"],
    opacity: Float[Array, "n_freq-1 n_temp n_dens"],
    emissivity: Float[Array, "n_freq-1 n_temp n_dens"],
):
    """
    Builds an interpolant for a single opacity table

    Parameters
    ----------
    energy : Float[Array, "n_freq"]
        The energy boundaries of the opacity table in keV
    temperature : Float[Array, "n_temp"]
        The temperature values of the opacity table in keV
    density : Float[Array, "n_dens"]
        The density values of the opacity table in 1/cm^3
    opacity : Float[Array, "n_freq-1 n_temp n_dens"]
        The Rosseland mean opacity values of the opacity table in 1/cm
    emissivity : Float[Array, "n_freq-1 n_temp n_dens"]
        The emissivity values of the opacity table in 1/cm

    Returns
    -------
    interpolant : Callable[[Float[Array, "*elements"], Float[Array, "*elements"], Float[Array, "*elements"]], tuple[Float[Array, "*elements"], Float[Array, "*elements"]]
        The interpolator function for the opacity table that takes density and temperature as input and returns the opacity and emissivity.
    shape : tuple[int, int]
        The shape of the opacity table
    energy : Float[Array, "n_freq"]
        The energy boundaries of the opacity table in keV
    """
    table_rho = jnp.log10(density)
    table_te = jnp.log10(temperature)
    table_ross_opacity = jnp.log10(opacity)
    hnu_kT = energy[:, None, None] / temperature[None, None, :]
    planck_eval = planck_integral(hnu_kT)
    planck_eval = planck_eval[1:, ...] - planck_eval[:-1, ...]
    emission_density = EMISS_CONST * (
        emissivity[:, :, :] * planck_eval * temperature**4
    )

    table_emiss_opacity = jnp.log10(emission_density + 1e-100)
    all_opacities = jnp.stack((table_ross_opacity, table_emiss_opacity), axis=0)

    @jax.jit
    def interpolant(
        density: Float[Array, "*elements"], temperature: Float[Array, "*elements"]
    ) -> Float[Array, "2 n_freq-1 *elements"]:
        ndens = jnp.log10(density.ravel())
        temp = jnp.log10(temperature.ravel())

        dens_index = jnp.clip(
            jnp.searchsorted(table_rho, ndens, side="right"), 1, len(table_rho) - 1
        )
        te_index = jnp.clip(
            jnp.searchsorted(table_te, temp, side="right"), 1, len(table_te) - 1
        )

        dT2 = (temp - table_te[te_index - 1]) / (
            table_te[te_index] - table_te[te_index - 1]
        )
        dT1 = (table_te[te_index] - temp) / (
            table_te[te_index] - table_te[te_index - 1]
        )

        dRho2 = (ndens - table_rho[dens_index - 1]) / (
            table_rho[dens_index] - table_rho[dens_index - 1]
        )
        dRho1 = (table_rho[dens_index] - ndens) / (
            table_rho[dens_index] - table_rho[dens_index - 1]
        )

        f11 = all_opacities[..., dens_index - 1, te_index - 1]
        f21 = all_opacities[..., dens_index, te_index - 1]
        f12 = all_opacities[..., dens_index - 1, te_index]
        f22 = all_opacities[..., dens_index, te_index]

        return 10 ** (
            dRho1 * dT1 * f11
            + dRho2 * dT1 * f21
            + dRho1 * dT2 * f12
            + dRho2 * dT2 * f22
        ).reshape(all_opacities.shape[0:2] + density.shape)

    return interpolant, all_opacities.shape[0:2], energy


@partial(jax.jit, static_argnames=("include_opacity", "ds"))
def raytrace_step(
    i: int,
    I0: Float[Array, "*pixels"],
    opacity: Float[Array, "*pixels nray"],
    emissivity: Float[Array, "*pxiels nray"],
    ds: float,
    *,
    include_opacity: bool = True,
) -> Float[Array, "*pixels"]:
    """
    Raytrace a single step of the radiograph

    Parameters
    ----------
    i : int
        The spatial index along the ray
    I0 : Float[Array, "*batch"]
        The intensity at the previous step for each pixel and energy in the radiograph
    opacity : Float[Array, "*pixels nray"]
        The opacity at each spatial index for each pixel and energy in the radiograph
    emissivity : Float[Array, "*pixels nray"]
        The emissivity at each spatial index for each pixel and energy in the radiograph
    ds : float
        The step size in cm
    include_opacity : bool
        If True, include opacity in the calculation. If False, only include emissivity.

    Returns
    -------
    Float[Array, "*pixels"]
        The intensity at the current step for each pixel and energy in the radiograph
    """
    if include_opacity:
        efac = jnp.exp(-opacity[..., i] * ds)
        return I0 * efac + (emissivity[..., i] / opacity[..., i]) * (1 - efac)
    return I0 + emissivity[..., i] * ds


@partial(jax.jit, static_argnames=("include_opacity", "ds"))
def raytrace_lines(
    opacity: Float[Array, "*pixels nray"],
    emissivity: Float[Array, "*pixels nray"],
    ds: float,
    *,
    include_opacity: bool = True,
) -> Float[Array, "*pixels"]:
    """
    Raytrace along a set of rays to generate a radiograph

    Parameters
    ----------
    opacity : Float[Array, "*pixels nray"]
        The opacity along each ray for each pixel and energy in the radiograph
    emissivity : Float[Array, "*pixels nray"]
        The emissivity along each ray  for each pixel and energy in the radiograph
    ds : float
        The step size in cm
    include_opacity : bool
        If True, include opacity in the calculation. If False, only include emissivity.

    Returns
    -------
    Float[Array, "*pixels"]
        The intensity at the end of the ray for each pixel and energy in the radiograph
    """
    I0 = jnp.zeros(opacity.shape[0:-1])
    step_function = Partial(
        raytrace_step,
        ds=ds,
        opacity=opacity,
        emissivity=emissivity,
        include_opacity=include_opacity,
    )
    return jax.lax.fori_loop(0, opacity.shape[-1], step_function, I0)


def build_projector_1d(
    image_axis: Float[Array, "n_r"],
    opacity_interpolator: Callable[
        [
            Float[Array, "*elements"],
            Float[Array, "*elements"],
            Float[Array, "*elements"],
        ],
        tuple[Float[Array, "*elements"], Float[Array, "*elements"]],
    ],
    photon_energies: Float[Array, "n_freq"],
    *,
    time_integrate: bool = False,
    include_opacity: bool = True,
) -> Callable[
    [
        Float[Array, "npnts_in"],
        Float[Array, "npnts_in"],
        Float[Array, "npnts_in"],
        Float[Array, "npnts_in"],
    ],
    tuple[Float[Array, ...], Float[Array, ...]],
]:
    """
    Builds a projector function for a 1D radial lineout of a radiograph from a 1D plasma input

    Parameters
    ----------
    image_axis : Float[Array, "n_r"]
        The axis of the synthetic radiograph image in cm
    opacity_interpolator : Callable[[Float[Array, "*elements"], Float[Array, "*elements"], Float[Array, "*elements"]], tuple[Float[Array, "*elements"], Float[Array, "*elements"]]
        The interpolator function for the opacity tables that takes density, temperature, and material fractions as input and returns the emission density and transport opacity.
    photon_energies : Float[Array, "n_freq"]
        The photon energies of the opacity table in keV
    time_integrate : bool
        If True, the projector will integrate over time. If False, the projector will return a time-resolved radiograph.
    include_opacity : bool
        If True, the projector will include opacity in the calculation. If False, the projector will only include emissivity.
    """
    projection_axis = jnp.linspace(
        -image_axis.max(), image_axis.max(), 2 * image_axis.size + 1
    )
    x, y = jnp.meshgrid(image_axis, projection_axis, indexing="ij")
    r_interp = jnp.sqrt(x**2 + y**2)
    ds = jnp.mean(jnp.diff(projection_axis))

    @jax.jit
    def raytrace_one_time_step(
        time_index: int,
        radius: Float[Array, "npnts_in"],
        density: Float[Array, "npnts_in"],
        temperature: Float[Array, "npnts_in"],
        material_fractions: Float[Array, "npnts_in"],
    ) -> Float[Array, ...]:
        r = radius[time_index]
        dens = density[time_index]
        temp = temperature[time_index]
        fracs = material_fractions[:, time_index]
        emission_density, transport_opacity = opacity_interpolator(
            dens,
            temp,
            fracs,
        )

        i = jnp.clip(jnp.searchsorted(r, r_interp, side="right"), 1, r.shape[0] - 1)
        dR = jnp.clip((r[i] - r_interp) / (r[i] - r[i - 1]), 0, 1)
        em_interp = dR * emission_density[:, i - 1] + (1 - dR) * emission_density[:, i]
        op_interp = (
            dR * transport_opacity[:, i - 1] + (1 - dR) * transport_opacity[:, i]
        )
        return raytrace_lines(op_interp, em_interp, ds, include_opacity=include_opacity)

    @jax.jit
    def raytrace_time_integrate(
        radius: Float[Array, "npnts_in"],
        density: Float[Array, "npnts_in"],
        temperature: Float[Array, "npnts_in"],
        material_fractions: Float[Array, "npnts_in"],
    ) -> tuple[Float[Array, ...], Float[Array, ...]]:
        image = jnp.zeros((len(photon_energies), len(image_axis)))
        return photon_energies, jax.lax.fori_loop(
            0,
            len(radius),
            lambda i, image: image
            + raytrace_one_time_step(
                i, radius, density, temperature, material_fractions
            ),
            image,
        )

    @jax.jit
    def raytrace_time_resolved(
        radius: Float[Array, "npnts_in"],
        density: Float[Array, "npnts_in"],
        temperature: Float[Array, "npnts_in"],
        material_fractions: Float[Array, "npnts_in"],
    ) -> tuple[Float[Array, ...], Float[Array, ...]]:
        image = jnp.array(
            [
                raytrace_one_time_step(
                    i, radius, density, temperature, material_fractions
                )
                for i in range(len(radius))
            ]
        )
        return photon_energies, image

    if time_integrate:
        return raytrace_time_integrate
    return raytrace_time_resolved


def build_projector_3d(
    image_axis: Float[Array, "n_r"],
    theta_detector: float,
    phi_detector: float,
    opacity_interpolator: Callable[
        [
            Float[Array, "*elements"],
            Float[Array, "*elements"],
            Float[Array, "*elements"],
        ],
        tuple[Float[Array, "*elements"], Float[Array, "*elements"]],
    ],
    photon_energies: Float[Array, "n_freq"],
    *,
    time_integrate: bool = False,
    include_opacity: bool = True,
) -> Callable[
    [
        Float[Array, "*npnts_in"],
        Float[Array, "*npnts_in"],
        Float[Array, "*npnts_in"],
        Float[Array, "*npnts_in"],
    ],
    tuple[Float[Array, ...], Float[Array, ...]],
]:
    image_normal = SphericalVector(1.0, theta_detector, phi_detector).to_cartesian()
    projection_basis = image_plane_basis(image_normal)
    hx, hy, hz = jnp.meshgrid(image_axis, image_axis, image_axis)
    hx = hx[:, :, :, None] * projection_basis[0].to_cartesian().normalized.data
    hy = hy[:, :, :, None] * projection_basis[1].to_cartesian().normalized.data
    hz = hz[:, :, :, None] * image_normal.to_cartesian().normalized.data
    hxyz = hx + hy + hz
    ds = jnp.mean(jnp.diff(image_axis))

    @jax.jit
    def vmapped_raytrace(xyz_eval, r, t, p, emission_density, transport_opacity, ds):
        hr = jnp.linalg.norm(xyz_eval, axis=-1)
        ht = jnp.arccos(xyz_eval[..., 2] / hr)
        hp = jnp.arctan2(xyz_eval[..., 1], xyz_eval[..., 0])
        hp = jnp.where(hp < 0, hp + 2 * jnp.pi, hp)

        i = jnp.clip(jnp.searchsorted(p, hp, side="right"), 1, p.shape[0] - 1)
        j = jnp.clip(jnp.searchsorted(t, ht, side="right"), 1, t.shape[0] - 1)
        k = jnp.clip(jnp.searchsorted(r, hr, side="right"), 1, r.shape[0] - 1)
        x0 = p[i - 1]
        x1 = p[i]
        y0 = t[j - 1]
        y1 = t[j]
        z0 = r[k - 1]
        z1 = r[k]
        dxi = 1 / (x1 - x0)
        dyi = 1 / (y1 - y0)
        dzi = 1 / (z1 - z0)

        tx = jnp.clip(jnp.array([x1 - hp, hp - x0]) * dxi, 0, 1)
        ty = jnp.clip(jnp.array([y1 - ht, ht - y0]) * dyi, 0, 1)
        tz = jnp.clip(jnp.array([z1 - hr, hr - z0]) * dzi, 0, 1)

        f000 = emission_density[:, i - 1, j - 1, k - 1]
        f001 = emission_density[:, i - 1, j - 1, k]
        f010 = emission_density[:, i - 1, j, k - 1]
        f100 = emission_density[:, i, j - 1, k - 1]
        f110 = emission_density[:, i, j, k - 1]
        f011 = emission_density[:, i - 1, j, k]
        f101 = emission_density[:, i, j - 1, k]
        f111 = emission_density[:, i, j, k]
        F = jnp.array([[[f000, f001], [f010, f011]], [[f100, f101], [f110, f111]]])
        emission = jnp.einsum("lij...k,lk,ik,jk->...k", F, tx, ty, tz)
        f000 = transport_opacity[:, i - 1, j - 1, k - 1]
        f001 = transport_opacity[:, i - 1, j - 1, k]
        f010 = transport_opacity[:, i - 1, j, k - 1]
        f100 = transport_opacity[:, i, j - 1, k - 1]
        f110 = transport_opacity[:, i, j, k - 1]
        f011 = transport_opacity[:, i - 1, j, k]
        f101 = transport_opacity[:, i, j - 1, k]
        f111 = transport_opacity[:, i, j, k]
        F = jnp.array([[[f000, f001], [f010, f011]], [[f100, f101], [f110, f111]]])
        absorption = jnp.einsum("lij...k,lk,ik,jk->...k", F, tx, ty, tz)
        return raytrace_lines(absorption, emission, ds, include_opacity=include_opacity)

    @jax.jit
    def raytrace_one_time_step(
        time_index: int,
        radius: Float[Array, "*npnts_in"],
        theta: Float[Array, "*npnts_in"],
        phi: Float[Array, "*npnts_in"],
        density: Float[Array, "*npnts_in"],
        temperature: Float[Array, "*npnts_in"],
        material_fractions: Float[Array, "nmats *npnts_in"],
    ):
        r = radius[time_index]
        t = theta[time_index]
        p = phi[time_index]
        dens = density[time_index]
        temp = temperature[time_index]
        fracs = material_fractions[:, time_index]
        emission_density, transport_opacity = opacity_interpolator(dens, temp, fracs)
        return jax.vmap(
            jax.vmap(vmapped_raytrace, in_axes=(0, None, None, None, None, None, None)),
            in_axes=(1, None, None, None, None, None, None),
        )(hxyz, r, t, p, emission_density, transport_opacity, ds)

    @jax.jit
    def raytrace_time_integrated(
        radius: Float[Array, "*npnts_in"],
        theta: Float[Array, "*npnts_in"],
        phi: Float[Array, "*npnts_in"],
        density: Float[Array, "*npnts_in"],
        temperature: Float[Array, "*npnts_in"],
        material_fractions: Float[Array, "nmats *npnts_in"],
    ):
        image = jnp.zeros(
            (
                len(image_axis),
                len(image_axis),
                len(photon_energies),
            )
        )
        return photon_energies, jax.lax.fori_loop(
            0,
            len(radius),
            lambda i, image: image
            + raytrace_one_time_step(
                i, radius, theta, phi, density, temperature, material_fractions
            ),
            image,
        )

    @jax.jit
    def raytrace_time_resolved(
        radius: Float[Array, "*npnts_in"],
        theta: Float[Array, "*npnts_in"],
        phi: Float[Array, "*npnts_in"],
        density: Float[Array, "*npnts_in"],
        temperature: Float[Array, "*npnts_in"],
        material_fractions: Float[Array, "nmats *npnts_in"],
    ):
        image = jnp.zeros(
            (
                len(radius),
                len(image_axis),
                len(image_axis),
                len(photon_energies),
            )
        )
        return photon_energies, jax.lax.fori_loop(
            0,
            len(radius),
            lambda i, image: image[i]
            + raytrace_one_time_step(
                i, radius, theta, phi, density, temperature, material_fractions
            ),
            image,
        )

    if time_integrate:
        return raytrace_time_integrated
    return raytrace_time_resolved
