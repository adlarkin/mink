"""Tests for pose_constraint_task.py."""

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import SE3, SO3, Configuration
from mink.exceptions import TargetNotSet, TaskDefinitionError
from mink.tasks import PoseConstraintTask, RelativeFrameTask


class TestPoseConstraintTask(absltest.TestCase):
    """Test consistency of the pose constraint task."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("g1_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)

        np.random.seed(42)
        self.T_wt = SE3.sample_uniform()

    def test_cost_correctly_broadcast(self):
        task = PoseConstraintTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="torso_link",
            root_type="body",
            position_cost=1.0,
            orientation_cost=5.0,
        )
        np.testing.assert_array_equal(task.cost, np.array([1, 1, 1, 5, 5, 5]))

        task = PoseConstraintTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="torso_link",
            root_type="body",
            position_cost=[1.0, 2.0, 3.0],
            orientation_cost=[5.0, 6.0, 7.0],
        )
        np.testing.assert_array_equal(task.cost, np.array([1, 2, 3, 5, 6, 7]))

    def test_task_raises_error_if_cost_dim_invalid(self):
        with self.assertRaises(TaskDefinitionError):
            PoseConstraintTask(
                frame_name="pelvis",
                frame_type="body",
                root_name="torso_link",
                root_type="body",
                position_cost=[1.0, 2.0],
                orientation_cost=2.0,
            )
        with self.assertRaises(TaskDefinitionError):
            PoseConstraintTask(
                frame_name="pelvis",
                frame_type="body",
                root_name="torso_link",
                root_type="body",
                position_cost=7.0,
                orientation_cost=[2.0, 5.0],
            )

    def test_task_raises_error_if_cost_negative(self):
        with self.assertRaises(TaskDefinitionError):
            PoseConstraintTask(
                frame_name="pelvis",
                frame_type="body",
                root_name="torso_link",
                root_type="body",
                position_cost=1.0,
                orientation_cost=-1.0,
            )
        with self.assertRaises(TaskDefinitionError):
            PoseConstraintTask(
                frame_name="pelvis",
                frame_type="body",
                root_name="torso_link",
                root_type="body",
                position_cost=[-1.0, -1.0, -1.0],
                orientation_cost=[1, 2, 3],
            )

    def test_error_without_target(self):
        task = PoseConstraintTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="torso_link",
            root_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        with self.assertRaises(TargetNotSet):
            task.compute_error(self.configuration)

    def test_jacobian_without_target(self):
        task = PoseConstraintTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="torso_link",
            root_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        with self.assertRaises(TargetNotSet):
            task.compute_jacobian(self.configuration)

    def test_set_target_from_configuration(self):
        task = PoseConstraintTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="torso_link",
            root_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
        )
        task.set_target_from_configuration(self.configuration)

        # The target should not have been clamped since the task's limits are all
        # +- infinity.
        pose = self.configuration.get_transform("pelvis", "body", "torso_link", "body")
        self.assertEqual(task.transform_target_to_root, pose)

    def test_matches_relative_frame_task(self):
        # A PoseConstraintTask with the default limits (+- infinity) should exhibit
        # the same behavior as a RelativeFrameTask.
        constraint_task = PoseConstraintTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="world",
            root_type="body",
            position_cost=1.0,
            orientation_cost=5.0,
        )
        constraint_task.set_target(self.T_wt)

        relative_task = RelativeFrameTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="world",
            root_type="body",
            position_cost=1.0,
            orientation_cost=5.0,
        )
        relative_task.set_target(self.T_wt)

        np.testing.assert_allclose(
            constraint_task.compute_error(self.configuration),
            relative_task.compute_error(self.configuration),
        )
        np.testing.assert_allclose(
            constraint_task.compute_jacobian(self.configuration),
            relative_task.compute_jacobian(self.configuration),
        )

    def test_task_enforces_translation_limit(self):
        x_limits = (-1.0, 1.0)

        task = PoseConstraintTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="world",
            root_type="body",
            position_cost=1.0,
            orientation_cost=5.0,
            x_translation=x_limits,
        )

        # Make sure the initial transform is within the pose limits.
        transform_frame_to_root = self.configuration.get_transform(
            "pelvis", "body", "world", "body"
        )
        initial_x_translation = transform_frame_to_root.translation()[0]
        self.assertGreaterEqual(initial_x_translation, x_limits[0])
        self.assertLessEqual(initial_x_translation, x_limits[1])

        # Define a target that violates the x-axis translation limit.
        target_outside_translation_limit = transform_frame_to_root.copy()
        target_outside_translation_limit.translation()[0] = x_limits[1] + 1.0

        # The target should be clamped to enforce the translation limit.
        task.set_target(target_outside_translation_limit)
        self.assertAlmostEqual(
            task.transform_target_to_root.translation()[0], x_limits[1]
        )

        err = task.compute_error(self.configuration)
        np.testing.assert_allclose(
            err, np.array([-x_limits[1], 0.0, 0.0, 0.0, 0.0, 0.0])
        )

    def test_task_enforces_rotation_limit(self):
        roll_limits = (-np.pi / 2, np.pi / 2)

        task = PoseConstraintTask(
            frame_name="pelvis",
            frame_type="body",
            root_name="world",
            root_type="body",
            position_cost=1.0,
            orientation_cost=5.0,
            roll=roll_limits,
        )

        # Make sure the initial transform is within the pose limits.
        transform_frame_to_root = self.configuration.get_transform(
            "pelvis", "body", "world", "body"
        )
        initial_roll = transform_frame_to_root.rotation().as_rpy_radians().roll
        self.assertGreaterEqual(initial_roll, roll_limits[0])
        self.assertLessEqual(initial_roll, roll_limits[1])

        # Define a target that violates the roll limit.
        rpy = transform_frame_to_root.rotation().as_rpy_radians()
        rotation = SO3.from_rpy_radians(roll_limits[1] + 1.0, rpy.pitch, rpy.yaw)
        target_outside_rotation_limit = SE3.from_rotation_and_translation(
            rotation=rotation, translation=transform_frame_to_root.translation()
        )

        # The target should be clamped to enforce the roll limit.
        task.set_target(target_outside_rotation_limit)
        self.assertAlmostEqual(
            task.transform_target_to_root.rotation().as_rpy_radians().roll,
            roll_limits[1],
        )

        err = task.compute_error(self.configuration)
        np.testing.assert_allclose(
            err, np.array([0.0, 0.0, 0.0, -roll_limits[1], 0.0, 0.0])
        )


if __name__ == "__main__":
    absltest.main()
