"""
Copied and adapted from https://github.com/mila-iqia/babyai.
Levels described in the Baby AI ICLR 2019 submission, with the `Go to` instruction.
"""

from __future__ import annotations

import numpy as np
from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import (
    AfterInstr,
    BeforeInstr,
    GoToInstr,
    ObjDesc,
)


class GoToLocalPretraining(RoomGridLevel):
    def __init__(self, room_size=8, num_dists=7, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        color = np.random.choice(["red", "green", "blue"])
        obj_type = np.random.choice(["ball", "box"])
        obj, _ = self.add_object(0, 0, obj_type, color)
        self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


def goto_local_class(object, color):
    class GoToLocalTemplate(RoomGridLevel):
        def __init__(self, room_size=8, num_dists=7, **kwargs):
            self.num_dists = num_dists
            super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

        def gen_mission(self):
            self.place_agent()
            obj, _ = self.add_object(0, 0, object, color)
            self.add_distractors(num_distractors=self.num_dists, all_unique=False)
            self.check_objs_reachable()
            self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

    return GoToLocalTemplate


def goto_seq_class(o1, c1, o2, c2):
    class GoToLocalSeqTemplate(LevelGen):
        def __init__(self, room_size=20, num_rows=1, num_cols=1, num_dists=7, **kwargs):
            super().__init__(
                room_size=room_size,
                num_rows=num_rows,
                num_cols=num_cols,
                num_dists=num_dists,
                action_kinds=["goto"],
                locked_room_prob=0,
                locations=False,
                unblocking=False,
                **kwargs,
            )

        def gen_mission(self):
            if self._rand_float(0, 1) < self.locked_room_prob:
                self.add_locked_room()

            self.connect_all()

            obj1, _ = self.add_object(0, 0, o1, c1)
            obj2, _ = self.add_object(0, 0, o2, c2)
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

            instr_a = GoToInstr(ObjDesc(obj1.type, obj1.color))
            instr_b = GoToInstr(ObjDesc(obj2.type, obj2.color))

            kind = self._rand_elem(["before", "after"])
            if kind == "before":
                self.instrs = BeforeInstr(instr_a, instr_b)
            elif kind == "after":
                self.instrs = AfterInstr(instr_b, instr_a)

    return GoToLocalSeqTemplate


GoToLocalRedBall = goto_local_class("ball", "red")
GoToLocalBlueBall = goto_local_class("ball", "blue")
GoToLocalGreenBall = goto_local_class("ball", "green")
GoToLocalPurpleBall = goto_local_class("ball", "purple")
GoToLocalRedBox = goto_local_class("box", "red")
GoToLocalBlueBox = goto_local_class("box", "blue")
GoToLocalGreenBox = goto_local_class("box", "green")
GoToLocalPurpleBox = goto_local_class("box", "purple")
GoToLocalRedKey = goto_local_class("key", "red")
GoToLocalBlueKey = goto_local_class("key", "blue")
GoToLocalGreenKey = goto_local_class("key", "green")
GoToLocalPurpleKey = goto_local_class("key", "purple")


GoToRedBallBlueBox = goto_seq_class("ball", "red", "box", "blue")
GoToPurpleBallGreenKey = goto_seq_class("ball", "purple", "box", "purple")
