"""Joint velocity limit."""

from typing import List, Mapping, Optional

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..constants import dof_width
from ..exceptions import LimitDefinitionError
from .limit import Constraint, Limit


class VelocityLimit(Limit):
    """Inequality constraint on joint velocities in a robot model.

    Floating base joints are ignored.

    Attributes:
        indices: Tangent indices corresponding to velocity-limited joints. Shape (nb,).
        limit: Maximum allowed velocity magnitude for velocity-limited joints, in
            [m]/[s] for slide joints and [rad]/[s] for hinge joints. Shape (nb,).
        projection_matrix: Projection from tangent space to subspace with
            velocity-limited joints. Shape (nb, nv) where nb is the dimension of the
            velocity-limited subspace and nv is the dimension of the tangent space.
    """

    indices: np.ndarray
    limit: np.ndarray
    projection_matrix: Optional[np.ndarray]

    def __init__(
        self,
        model: mujoco.MjModel,
        velocities: Mapping[str, npt.ArrayLike] = {},
    ):
        """Initialize velocity limits.

        Args:
            model: MuJoCo model.
            velocities: Dictionary mapping joint name to maximum allowed magnitude in
                [m]/[s] for slide joints and [rad]/[s] for hinge joints.
        """
        limit_list: List[float] = []
        index_list: List[int] = []
        for joint_name, max_vel in velocities.items():
            jid = model.joint(joint_name).id
            jnt_type = model.jnt_type[jid]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                raise LimitDefinitionError(f"Free joint {joint_name} is not supported")
            vadr = model.jnt_dofadr[jid]
            vdim = dof_width(jnt_type)
            max_vel = np.atleast_1d(max_vel)
            if max_vel.shape != (vdim,):
                raise LimitDefinitionError(
                    f"Joint {joint_name} must have a limit of shape ({vdim},). "
                    f"Got: {max_vel.shape}"
                )
            index_list.extend(range(vadr, vadr + vdim))
            limit_list.extend(max_vel.tolist())  # type: ignore

        self.indices = np.array(index_list)
        self.indices.setflags(write=False)
        self.limit = np.array(limit_list)
        self.limit.setflags(write=False)

        nb = len(self.indices)
        self.projection_matrix = np.eye(model.nv)[self.indices] if nb > 0 else None

    def compute_qp_inequalities(
        self, configuration: Configuration, dt: float
    ) -> Constraint:
        r"""Compute the configuration-dependent joint velocity limits.

        The limits are defined as:

        .. math::

            -v_{\text{max}} \cdot dt \leq \Delta q \leq v_{\text{max}} \cdot dt

        where :math:`v_{max} \in {\cal T}` is the robot's velocity limit
        vector and :math:`\Delta q \in T_q({\cal C})` is the displacement in the
        tangent space at :math:`q`. See the :ref:`derivations` section for
        more information.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Integration timestep in [s].

        Returns:
            Pair :math:`(G, h)` representing the inequality constraint as
            :math:`G \Delta q \leq h`, or ``None`` if there is no limit. G has
            shape (2nb, nv) and h has shape (2nb,).
        """
        del configuration  # Unused.
        if self.projection_matrix is None:
            return Constraint()
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([dt * self.limit, dt * self.limit])
        return Constraint(G=G, h=h)
