import numpy as np
from gymnasium import spaces
from crafter import Env

id_to_item = [0] * 19
import itertools

dummyenv = Env()
for name, ind in itertools.chain(dummyenv._world._mat_ids.items(), dummyenv._sem_view._obj_ids.items()):
    name = str(name)[str(name).find("objects.") + len("objects.") : -2].lower() if "objects." in str(name) else str(name)
    id_to_item[ind] = name
player_idx = id_to_item.index("player")
del dummyenv

vitals = [
    "health",
    "food",
    "drink",
    "energy",
]

rot = np.array([[0, -1], [1, 0]])
directions = ["front", "right", "back", "left"]


def describe_inventory(info):
    result = ""

    status_str = "Your status: {}".format(", ".join(["{}: {}/9".format(v, info["inventory"][v]) for v in vitals]))
    result += status_str + ". "

    inventory_str = ", ".join(["{} {}".format(num, i) for i, num in info["inventory"].items() if i not in vitals and num != 0])
    inventory_str = "Your have {}".format(inventory_str) if inventory_str else "You have nothing in your inventory."
    result += inventory_str  # + "\n\n"

    return result.strip()


REF = np.array([0, 1])


def rotation_matrix(v1, v2):
    dot = np.dot(v1, v2)
    cross = np.cross(v1, v2)
    rotation_matrix = np.array([[dot, -cross], [cross, dot]])
    return rotation_matrix


def describe_loc(ref, P):
    desc = []
    if ref[1] > P[1]:
        desc.append("up")
    elif ref[1] < P[1]:
        desc.append("down")
    if ref[0] > P[0]:
        desc.append("left")
    elif ref[0] < P[0]:
        desc.append("right")

    return "-".join(desc)


def describe_env(info):
    assert info["semantic"][info["player_pos"][0], info["player_pos"][1]] == player_idx
    semantic = info["semantic"][info["player_pos"][0] - info["view"][0] // 2 : info["player_pos"][0] + info["view"][0] // 2 + 1, info["player_pos"][1] - info["view"][1] // 2 + 1 : info["player_pos"][1] + info["view"][1] // 2]
    center = np.array([info["view"][0] // 2, info["view"][1] // 2 - 1])
    result = ""
    x = np.arange(semantic.shape[1])
    y = np.arange(semantic.shape[0])
    x1, y1 = np.meshgrid(x, y)
    loc = np.stack((y1, x1), axis=-1)
    dist = np.absolute(center - loc).sum(axis=-1)
    obj_info_list = []

    facing = info["player_facing"]
    target = (center[0] + facing[0], center[1] + facing[1])
    target = id_to_item[semantic[target]]
    obs = "You face {} at your front.".format(target, describe_loc(np.array([0, 0]), facing))

    for idx in np.unique(semantic):
        if idx == player_idx:
            continue

        smallest = np.unravel_index(np.argmin(np.where(semantic == idx, dist, np.inf)), semantic.shape)
        obj_info_list.append((id_to_item[idx], dist[smallest], describe_loc(np.array([0, 0]), smallest - center)))

    if len(obj_info_list) > 0:
        status_str = "You see: {}.".format(", ".join(["{} at {} steps to your {}".format(name, dist, loc) for name, dist, loc in obj_info_list]))
    else:
        status_str = "You see nothing away from you."
    result += status_str
    result += obs.strip()

    return result.strip()


def describe_status(info):
    if info["sleeping"]:
        return "You are sleeping, and will not be able take actions until energy is full."
    elif info["dead"]:
        return "You died."
    else:
        return ""


def describe_frame(info):
    try:
        result = ""
        result += describe_status(info)
        result += " "
        result += describe_env(info)
        result += " "
        result += describe_inventory(info)

        return result.strip()
    except:
        return "Error, you are out of the map."


class CrafterText(Env):
    def __init__(self, area=(64, 64), view=(9, 9), size=(64, 64), reward=True, length=10000, seed=None, task=None):
        super().__init__(area, view, size, reward, length, seed, task)

    @property
    def observation_space(self):
        return spaces.Dict(
            {
                "image": spaces.Box(0, 255, tuple(self._size) + (3,), np.uint8),
                "description": spaces.Text(4096),
            }
        )

    def text_obs(self, obs, info):
        return dict(image=obs, description=describe_frame(info))

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        obs, reward, terminated, truncated, info = super().step(0)
        return self.text_obs(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self.text_obs(obs, info), reward, terminated, truncated, info
