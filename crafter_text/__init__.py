import crafter
from text_env import CrafterText

import gymnasium as gym

gym.register(id="Crafter-Text-Reward", entry_point=CrafterText, max_episode_steps=10000)

gym.register(id="Crafter-Text-PlaceCraftingTable", entry_point=CrafterText, max_episode_steps=10000, kwargs={"task": "place_table"})
gym.register(id="Crafter-Text-MakeWoodPickAxe", entry_point=CrafterText, max_episode_steps=10000, kwargs={"task": "make_wood_pickaxe"})
gym.register(id="Crafter-Text-CollectDrink", entry_point=CrafterText, max_episode_steps=10000, kwargs={"task": "collect_drink"})
gym.register(id="Crafter-Text-EatCow", entry_point=CrafterText, max_episode_steps=10000, kwargs={"task": "eat_cow"})
