import gymnasium as gym

from .goto import *
from .pickup import *

gym.register("BabyAI-Text-GoToLocal-Pretraining-S20", entry_point=GoToLocalPretraining, kwargs=dict(room_size=20))

gym.register("BabyAI-Text-GoToLocal-RedBall-S20", entry_point=GoToLocalRedBall, kwargs=dict(room_size=20))
gym.register("BabyAI-Text-GoToLocal-PurpleBall-S20", entry_point=GoToLocalPurpleBall, kwargs=dict(room_size=20))
gym.register("BabyAI-Text-GoToLocal-RedKey-S20", entry_point=GoToLocalRedKey, kwargs=dict(room_size=20))
gym.register("BabyAI-Text-GoToLocal-PurpleKey-S20", entry_point=GoToLocalPurpleKey, kwargs=dict(room_size=20))

gym.register("BabyAI-Text-GoToLocal-RedBall-S23", entry_point=GoToLocalRedBall, kwargs=dict(room_size=23, num_dists=12))
gym.register("BabyAI-Text-GoToLocal-PurpleKey-S23", entry_point=GoToLocalPurpleKey, kwargs=dict(room_size=23, num_dists=12))

gym.register("BabyAI-Text-GoToLocal-RedBall-S25", entry_point=GoToLocalRedBall, kwargs=dict(room_size=25, num_dists=15))
gym.register("BabyAI-Text-GoToLocal-PurpleKey-S25", entry_point=GoToLocalPurpleKey, kwargs=dict(room_size=25, num_dists=15))

gym.register("BabyAI-Text-GoToLocal-RedBall-S28", entry_point=GoToLocalRedBall, kwargs=dict(room_size=28, num_dists=20))
gym.register("BabyAI-Text-GoToLocal-PurpleKey-S28", entry_point=GoToLocalPurpleKey, kwargs=dict(room_size=28, num_dists=20))

gym.register("BabyAI-Text-GoToLocal-RedBall-S30", entry_point=GoToLocalRedBall, kwargs=dict(room_size=30, num_dists=25))
gym.register("BabyAI-Text-GoToLocal-PurpleKey-S30", entry_point=GoToLocalPurpleKey, kwargs=dict(room_size=30, num_dists=25))

gym.register("BabyAI-Text-GoTo-RedBallBlueBox-S20", entry_point=GoToRedBallBlueBox, kwargs=dict(room_size=20))
gym.register("BabyAI-Text-GoTo-PurpleBallGreenKey-S20", entry_point=GoToPurpleBallGreenKey, kwargs=dict(room_size=20))

gym.register("BabyAI-Text-PickUp-RedBall-S20", entry_point=PickUpRedBall, kwargs=dict(room_size=20))
gym.register("BabyAI-Text-PickUp-PurpleKey-S20", entry_point=PickUpPurpleKey, kwargs=dict(room_size=20))
