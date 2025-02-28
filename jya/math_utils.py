from __future__ import annotations

from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np
import scipy.integrate

__all__ = [
    "CartesianVector",
    "SphericalVector",
    "image_plane_basis",
    "image_plane_projection",
    "planck_integral",
]


_x = np.linspace(start=0, stop=30, num=30_000)
_y = _x**3 / np.clip(np.exp(_x) - 1, a_min=1e-10, a_max=np.inf)
_y[0] = 0
_yintegral = (
    15
    * scipy.integrate.cumulative_trapezoid(initial=0, x=_x, y=_y)  # type: ignore
    / np.pi**4
)


def planck_integral(x: Float[Array, ...]) -> Float[Array, ...]:
    """
    Computes the Planck integral from 0 to x for the given x = hnu/kT.

    If x is None, returns 0. If x is an array, interpolates the integral values.

    :param x: Input value for the Planck integral to be computed for
    """
    return jnp.interp(x=x.ravel(), xp=_x, fp=_yintegral, left=0, right=1.0).reshape(
        x.shape
    )


class CartesianVector:
    """A container for a 3-D vector represented in Cartesian coordinates."""

    _data: Float[Array, ...]

    def __init__(
        self, x: Float[Array, "n"], y: Float[Array, "n"], z: Float[Array, "n"]
    ):
        x = jnp.atleast_1d(x)
        y = jnp.atleast_1d(y)
        z = jnp.atleast_1d(z)

        self._data = jnp.stack([x, y, z], axis=-1)

    @property
    def data(self):
        return self._data

    @property
    def normalized(self) -> Self:
        """Divides vector by magnitude for normalization."""
        return self / self.magnitude

    @classmethod
    def from_data(cls, data: Float[Array, ...]):
        """Create a Cartesian vector from data."""
        return cls(data[..., 0], data[..., 1], data[..., 2])

    @classmethod
    def from_spherical(cls, v: SphericalVector):
        """Create a Cartesian vector from a Spherical vector."""
        return cls(
            x=v.magnitude * jnp.sin(v.theta) * jnp.cos(v.phi),
            y=v.magnitude * jnp.sin(v.theta) * jnp.sin(v.phi),
            z=v.magnitude * jnp.cos(v.theta),
        )

    @classmethod
    def from_cartesian(cls, v: CartesianVector):
        """Create a Cartesian vector from a Cartesian vector."""
        return cls(v.x, v.y, v.z)

    def to_spherical(self):
        """Returns spherical vector in Cartesian coordinates."""
        return SphericalVector.from_cartesian(self)

    def to_cartesian(self):
        """Returns Cartesian vector in Cartesian coordinates."""
        return self

    @property
    def x(self):
        """Return the x component of the vector."""
        return self._data[..., 0]

    @property
    def y(self):
        """Return the y component of the vector."""
        return self._data[..., 1]

    @property
    def z(self):
        """Return the z component of the vector."""
        return self._data[..., 2]

    @property
    def magnitude(self):
        """Returns the magnitude of the vector."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __add__(self, other: CartesianVector | SphericalVector):
        return self.__class__(
            x=self.x + other.to_cartesian().x,
            y=self.y + other.to_cartesian().y,
            z=self.z + other.to_cartesian().z,
        )

    def __sub__(self, other: CartesianVector | SphericalVector):
        return self.__class__(
            x=self.x - other.to_cartesian().x,
            y=self.y - other.to_cartesian().y,
            z=self.z - other.to_cartesian().z,
        )

    def __mul__(self, other: float):
        return self.__class__(
            x=self.x * other,
            y=self.y * other,
            z=self.z * other,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__class__(
            x=self.x / other,
            y=self.y / other,
            z=self.z / other,
        )

    def flip(self):
        """Flip vector along x, y, and z axes."""
        return self.__class__(
            -self.x,
            -self.y,
            -self.z,
        )

    def central_angle(self, other: SphericalVector | CartesianVector):
        """Gets the central angle between two vectors."""
        return central_angle(self.to_spherical(), other.to_spherical())

    def spherical_cosine(self, other: SphericalVector | CartesianVector):
        """Gets the cosine between two vectors using spherical law of cosines."""
        return spherical_cosine(self.to_spherical(), other.to_spherical())

    def cross_product(self, other: SphericalVector | CartesianVector):
        """Gets the cross product of two vectors."""
        return cross_product(self.to_cartesian(), other.to_cartesian()).to_cartesian()

    def dot_product(self, other: SphericalVector | CartesianVector):
        """Gets the dot product of two vectors"""
        return dot_product(self.to_cartesian(), other.to_cartesian())


class SphericalVector:
    """A container for a 3-D vector represented in spherical coordinates."""

    _data: Float[Array, ...]

    def __init__(
        self,
        magnitude: Float[Array, "n"],
        theta: Float[Array, "n"],
        phi: Float[Array, "n"],
    ):
        theta = jnp.atleast_1d(theta)
        phi = jnp.atleast_1d(phi)

        phi = jnp.where(phi < 0, 2 * np.pi + phi, phi)

        self._data = jnp.stack([magnitude, theta, phi], axis=-1)

    @classmethod
    def from_data(cls, data: Float[Array, ...]):
        """Create a Spherical vector from data."""
        return cls(data[..., 0], data[..., 1], data[..., 2])

    @classmethod
    def from_cartesian(cls, v: CartesianVector):
        """Converts Cartesian vector to its equivalent spherical coordinates."""
        magnitude = v.magnitude
        theta = jnp.arccos(v.z / v.magnitude)
        phi = jnp.arctan2(v.y, v.x)
        phi = jnp.where(phi < 0, 2 * np.pi + phi, phi)

        return cls(
            magnitude=magnitude,
            theta=theta,
            phi=phi,
        )

    @property
    def normalized(self) -> Self:
        """Returns normalized version of vector."""
        return self / self.magnitude  # type: ignore

    @classmethod
    def from_spherical(cls, v: SphericalVector):
        """Creates an instance of this class from the spherical vector coordinates."""
        return cls(v.magnitude, v.theta, v.phi)

    def to_cartesian(self):
        """Converts spherical vector to Cartesian coordinates."""
        return CartesianVector.from_spherical(self)

    def to_spherical(self):
        return self

    @property
    def magnitude(self):
        """Returns magnitude from data."""
        return self._data[..., 0]

    @property
    def theta(self):
        """Returns theta from data."""
        return self._data[..., 1]

    @property
    def phi(self):
        """Return phi from data."""
        return self._data[..., 2]

    def flip(self):
        """Flips vector components along x, y, and z axes."""
        return self.__class__(
            self.magnitude,
            -self.theta + np.pi,
            self.phi + np.pi,
        )

    def rotate_on_sphere(self, theta, phi):
        return self.__class__(
            self.magnitude,
            self.theta + theta,
            self.phi + phi,
        )

    def central_angle(self, other: SphericalVector | CartesianVector):
        """Returns the central angle between two vectors."""
        return central_angle(self.to_spherical(), other.to_spherical())

    def spherical_cosine(self, other: SphericalVector | CartesianVector):
        """Returns the spherical cosine between two vectors."""
        return spherical_cosine(self.to_spherical(), other.to_spherical())

    def cross_product(self, other: SphericalVector | CartesianVector):
        """Returns the cross product of two vectors."""
        return cross_product(self.to_cartesian(), other.to_cartesian()).to_spherical()

    def dot_product(self, other: SphericalVector | CartesianVector):
        """Returns the dot product of two vectors."""
        return dot_product(self.to_cartesian(), other.to_cartesian())

    def __add__(self, other: CartesianVector | SphericalVector):
        return CartesianVector(
            x=self.to_cartesian().x + other.to_cartesian().x,
            y=self.to_cartesian().y + other.to_cartesian().y,
            z=self.to_cartesian().z + other.to_cartesian().z,
        ).to_spherical()

    def __sub__(self, other: CartesianVector | SphericalVector):
        return CartesianVector(
            x=self.to_cartesian().x - other.to_cartesian().x,
            y=self.to_cartesian().y - other.to_cartesian().y,
            z=self.to_cartesian().z - other.to_cartesian().z,
        )

    def __mul__(self, other: float):
        return CartesianVector(
            x=self.to_cartesian().x * other,
            y=self.to_cartesian().y * other,
            z=self.to_cartesian().z * other,
        ).to_spherical()

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return CartesianVector(
            x=self.to_cartesian().x / other,
            y=self.to_cartesian().y / other,
            z=self.to_cartesian().z / other,
        ).to_spherical()


def image_plane_basis(
    normal: SphericalVector | CartesianVector,
) -> tuple[SphericalVector, SphericalVector]:
    """
    Calculates the basis vectors for the image plane via normal vectors.

    :param normal: Normal vector, can be `SphericalVector` or `CartesianVector`

    :return: Tuple with two vectors containing basis vectors for the image plane
    """
    normal_sphr = normal.to_spherical()
    bx = SphericalVector(
        np.ones_like(normal_sphr.phi),
        np.pi / 2 + np.zeros_like(normal_sphr.phi),
        np.pi / 2 + normal_sphr.phi,
    )
    if normal_sphr.theta > np.pi / 2:
        by = SphericalVector(
            np.ones_like(normal_sphr.phi),
            normal_sphr.theta - np.pi / 2,
            normal_sphr.phi,
        )
    else:
        by = SphericalVector(
            np.ones_like(normal_sphr.phi),
            np.pi / 2 - normal_sphr.theta,
            normal_sphr.phi + np.pi,
        )
    return (bx, by)


def image_plane_projection(
    vector: SphericalVector | CartesianVector,
    basis: tuple[SphericalVector, SphericalVector],
    rotation_angle: float,
) -> CartesianVector:
    """
    Projects a vector onto an image plane.

    :param vector: Vector to be projected
    :param basis: Spherical vectors that define the plane
    :param rotation_angle: Angle of which the projected vector is rotated

    :return: Cartesian vector coordinated after projection and rotation
    """
    cosx = vector.spherical_cosine(basis[0])
    cosy = vector.spherical_cosine(basis[1])

    vx = vector.magnitude * cosx
    vy = vector.magnitude * cosy

    vxp = np.cos(rotation_angle) * (vx) - np.sin(rotation_angle) * (vy)  # type: ignore
    vyp = np.sin(rotation_angle) * (vx) + np.cos(rotation_angle) * (vy)  # type: ignore

    return CartesianVector(
        vxp,
        vyp,
        0 * vector.magnitude,
    )


def dot_product(
    v0: CartesianVector,
    v1: CartesianVector,
):
    """Returns the dot product between two Cartesian vectors."""
    return np.sum(v0.data * v1.data, axis=-1)


def cross_product(
    v0: CartesianVector,
    v1: CartesianVector,
) -> CartesianVector:
    """Returns the cross product of two Cartesian vectors."""
    return CartesianVector.from_data(np.cross(v0.data, v1.data))


def spherical_cosine(v0: SphericalVector, v1: SphericalVector):
    """Returns the spherical cosine between two spherical vectors."""
    return np.cos(v0.theta) * np.cos(v1.theta) + np.sin(v0.theta) * np.sin(
        v1.theta
    ) * np.cos(v1.phi - v0.phi)


def central_angle(v0: SphericalVector, v1: SphericalVector):
    """Returns the central angle between two spherical vectors."""
    return np.arccos(spherical_cosine(v0, v1))
