# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Planar Walker Domain."""
import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
from dm_control.mujoco.wrapper import mjbindings
from dm_env import specs
import numpy as np

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.93

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8

SUITE = containers.TaggedTasks()

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return resources.GetResource("tonic/environments/custom_environments/models/cassie-simulate/cassie.xml"), common.ASSETS

def randomize_limited_and_rotational_joints(physics, random=None):
    """Randomizes the positions of joints defined in the physics body.
    The following randomization rules apply:
      - Bounded joints (hinges or sliders) are sampled uniformly in the bounds.
      - Unbounded hinges are samples uniformly in [-pi, pi]
      - Quaternions for unlimited free joints and ball joints are sampled
        uniformly on the unit 3-sphere.
      - Quaternions for limited ball joints are sampled uniformly on a sector
        of the unit 3-sphere.
      - The linear degrees of freedom of free joints are not randomized.
    Args:
      physics: Instance of 'Physics' class that holds a loaded model.
      random: Optional instance of 'np.random.RandomState'. Defaults to the global
        NumPy random state.
    """
    random = random or np.random

    hinge = mjbindings.enums.mjtJoint.mjJNT_HINGE
    slide = mjbindings.enums.mjtJoint.mjJNT_SLIDE
    ball = mjbindings.enums.mjtJoint.mjJNT_BALL
    free = mjbindings.enums.mjtJoint.mjJNT_FREE

    qpos = physics.named.data.qpos

    for joint_id in range(physics.model.njnt):
      joint_name = physics.model.id2name(joint_id, 'joint')
      joint_type = physics.model.jnt_type[joint_id]
      is_limited = physics.model.jnt_limited[joint_id]
      range_min, range_max = physics.model.jnt_range[joint_id]

      if is_limited:
        if joint_type == hinge or joint_type == slide:
          qpos[joint_name] = random.uniform(range_min, range_max)

        elif joint_type == ball:
          qpos[joint_name] = random_limited_quaternion(random, range_max)

      else:
          continue

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Walker domain."""

  # def torso_upright(self):
  #   """Returns projection from z-axes of torso to the z-axes of world."""
  #   return self.named.data.xmat['Ostrich/pelvis', 'zz']

  # def torso_height(self):
  #   """Returns the height of the torso."""
  #   return self.named.data.xpos['Ostrich/pelvis', 'z']

  # def horizontal_velocity(self):
  #   """Returns the horizontal velocity of the center-of-mass."""
  #   return self.named.data.sensordata['torso_subtreelinvel'][0]

  def joint_angles(self):
    """Returns the state without global orientation or position."""
    return self.data.qpos[3:].copy()  # Skip the 3 DoFs of the free root joint.

  def orientations(self):
    """Returns planar orientations of all bodies."""
    """print("--->")
    print(self.named.data.xmat[1])
    print("##########")
    print(self.named.data.xmat[1:, ['xx', 'xz']]) """

    return self.named.data.xmat[1:, ['xx', 'xz']].ravel()


class Cassie(base.Task):
  """A planar walker task."""

  def __init__(self, move_speed, random=None):
    """Initializes an instance of `PlanarWalker`.

    Args:
      move_speed: A float. If this value is zero, reward is given simply for
        standing up. Otherwise this specifies a target horizontal velocity for
        the walking task.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._move_speed = move_speed
    super(Cassie, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    In 'standing' mode, use initial orientation and small velocities.
    In 'random' mode, randomize joint angles and let fall to the floor.

    Args:
      physics: An instance of `Physics`.

    """
    randomize_limited_and_rotational_joints(physics, self.random)
    super(Cassie, self).initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of body orientations, height and velocites."""
    obs = collections.OrderedDict()

    obs['orientations'] = physics.orientations()
    # obs['height'] = physics.torso_height()
    # obs['positions'] = physics.joint_angles() #
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    #print("physics.torso_height()  = " + str(physics.torso_height()) )
    # standing = rewards.tolerance(physics.torso_height(),
    #                              bounds=(_STAND_HEIGHT, float('inf')),
    #                              margin=_STAND_HEIGHT/2)

    # upright = (1 + physics.torso_upright()) / 2
    # stand_reward = (3*standing + upright) / 4
    # if self._move_speed == 0:
    #   return stand_reward
    # else:
    #   move_reward = rewards.tolerance(physics.horizontal_velocity(),
    #                                   bounds=(self._move_speed, float('inf')),
    #                                   margin=self._move_speed/2,
    #                                   value_at_margin=0.5,
    #                                   sigmoid='linear')
    #   return stand_reward * (5*move_reward + 1) / 6
    return 0
