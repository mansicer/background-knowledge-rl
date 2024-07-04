"""
Copied and adapted from https://github.com/mila-iqia/babyai.
Levels described in the Baby AI ICLR 2019 submission, with the `Go to` instruction.
"""
from __future__ import annotations

import numpy as np
from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import (
    ObjDesc,
    PickupInstr,
)


def pickup_local_class(object, color):
    class PickupLocalTemplate(LevelGen):
        def __init__(self, **kwargs):
            # We add many distractors to increase the probability
            # of ambiguous locations within the same room
            super().__init__(
                action_kinds=["pickup"],
                instr_kinds=["action"],
                num_rows=1,
                num_cols=1,
                num_dists=8,
                locked_room_prob=0,
                locations=False,
                unblocking=False,
                **kwargs,
            )

        def gen_mission(self):
            if self._rand_float(0, 1) < self.locked_room_prob:
                self.add_locked_room()

            self.connect_all()

            obj1, _ = self.add_object(0, 0, object, color)
            self.add_distractors(num_distractors=self.num_dists, all_unique=False)

            # The agent must be placed after all the object to respect constraints
            while True:
                self.place_agent()
                start_room = self.room_from_pos(*self.agent_pos)
                # Ensure that we are not placing the agent in the locked room
                if start_room is self.locked_room:
                    continue
                break

            # If no unblocking required, make sure all objects are
            # reachable without unblocking
            if not self.unblocking:
                self.check_objs_reachable()

            self.instrs = PickupInstr(ObjDesc(obj1.type, obj1.color))

    return PickupLocalTemplate


PickUpRedBall = pickup_local_class("ball", "red")
PickUpPurpleKey = pickup_local_class("key", "purple")
