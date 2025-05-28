"""Pose constraint task implementation."""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..exceptions import TargetNotSet, TaskDefinitionError
from ..lie import SE3, SO3
from .task import Task


class PoseConstraintTask(Task):
    """Regulate the pose of a frame within translation and rotation limits defined
    relative to another frame.

    Attributes:
        frame_name: Name of the frame to regulate, typically the name of body, geom
            or site in the robot model.
        frame_type: The frame type: `body`, `geom` or `site`.
        root_name: Name of the frame the task is relative to.
        root_type: The root frame type: `body`, `geom` or `site`.
        pose_limits: A 6x2 matrix that specifies the min/max allowed translation (xyz)
            and rotation (RPY) in the root frame.
        transform_target_to_root: Target pose in the root frame.

    Example:

    .. code-block:: python

        # Enforce the pose of a site named `target` to have a position along the x-axis
        # within [0.0, 1.0] and a roll within +- pi/2, relative to the world frame.
        task = PoseConstraintTask(
            frame_name="target",
            frame_type="site",
            root_name="world",
            root_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
            x_translation=(0.0, 1.0),
            roll=(-np.pi / 2, np.pi / 2),
        )
    """

    k: int = 6
    transform_target_to_root: Optional[SE3]

    def __init__(
        self,
        frame_name: str,
        frame_type: str,
        root_name: str,
        root_type: str,
        position_cost: npt.ArrayLike,
        orientation_cost: npt.ArrayLike,
        x_translation: Tuple[float, float] = (-np.inf, np.inf),
        y_translation: Tuple[float, float] = (-np.inf, np.inf),
        z_translation: Tuple[float, float] = (-np.inf, np.inf),
        roll: Tuple[float, float] = (-np.inf, np.inf),
        pitch: Tuple[float, float] = (-np.inf, np.inf),
        yaw: Tuple[float, float] = (-np.inf, np.inf),
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.root_name = root_name
        self.root_type = root_type
        self.transform_target_to_root = None

        self.pose_limits = np.array(
            [x_translation, y_translation, z_translation, roll, pitch, yaw]
        )
        self.set_position_cost(position_cost)
        self.set_orientation_cost(orientation_cost)

    def set_position_cost(self, position_cost: npt.ArrayLike) -> None:
        position_cost = np.atleast_1d(position_cost)
        if position_cost.ndim != 1 or position_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be a vector of shape "
                "1 (aka identical cost for all coordinates) or (3,) but got "
                f"{position_cost.shape}"
            )
        if not np.all(position_cost >= 0.0):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be >= 0"
            )
        self.cost[:3] = position_cost

    def set_orientation_cost(self, orientation_cost: npt.ArrayLike) -> None:
        orientation_cost = np.atleast_1d(orientation_cost)
        if orientation_cost.ndim != 1 or orientation_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} orientation cost should be a vector of "
                "shape 1 (aka identical cost for all coordinates) or (3,) but got "
                f"{orientation_cost.shape}"
            )
        if not np.all(orientation_cost >= 0.0):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be >= 0"
            )
        self.cost[3:] = orientation_cost

    def set_target(self, transform_target_to_root: SE3) -> None:
        """Set the target pose in the root frame. The target pose will be clamped
        within the translation and rotation limits.

        Args:
            transform_target_to_root: Transform from the task target frame to the
                root frame.
        """
        # The target and limits are defined in the root frame, so clamping can be done
        # directly. RPY must be extracted from the target pose since rotation limits
        # are represented as RPY.
        translation = transform_target_to_root.translation()
        rpy = transform_target_to_root.rotation().as_rpy_radians()
        constrained_translation_and_rpy = np.clip(
            [*translation, rpy.roll, rpy.pitch, rpy.yaw],
            self.pose_limits[:, 0],
            self.pose_limits[:, 1],
        )

        self.transform_target_to_root = SE3.from_rotation_and_translation(
            rotation=SO3.from_rpy_radians(*constrained_translation_and_rpy[3:]),
            translation=constrained_translation_and_rpy[:3],
        )

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target pose from a given robot configuration.

        Args:
            configuration: Robot configuration :math:`q`.
        """
        self.set_target(
            configuration.get_transform(
                self.frame_name,
                self.frame_type,
                self.root_name,
                self.root_type,
            )
        )

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self.transform_target_to_root is None:
            raise TargetNotSet(self.__class__.__name__)

        transform_frame_to_root = configuration.get_transform(
            self.frame_name,
            self.frame_type,
            self.root_name,
            self.root_type,
        )
        return transform_frame_to_root.rminus(self.transform_target_to_root)

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        if self.transform_target_to_root is None:
            raise TargetNotSet(self.__class__.__name__)

        jacobian_frame_in_frame = configuration.get_frame_jacobian(
            self.frame_name, self.frame_type
        )
        jacobian_root_in_root = configuration.get_frame_jacobian(
            self.root_name, self.root_type
        )

        transform_frame_to_root = configuration.get_transform(
            self.frame_name,
            self.frame_type,
            self.root_name,
            self.root_type,
        )
        transform_frame_to_target = (
            self.transform_target_to_root.inverse() @ transform_frame_to_root
        )

        return transform_frame_to_target.jlog() @ (
            jacobian_frame_in_frame
            - transform_frame_to_root.inverse().adjoint() @ jacobian_root_in_root
        )
