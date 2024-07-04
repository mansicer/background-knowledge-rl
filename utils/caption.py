from minigrid.core.actions import Actions


def minigrid_action_to_str(action):
    if action == Actions.left:
        return "turn left"
    elif action == Actions.right:
        return "turn right"
    elif action == Actions.forward:
        return "forward"
    elif action == Actions.pickup:
        return "pick up the front object"
    elif action == Actions.drop:
        return "drop your taken object"
    elif action == Actions.toggle:
        return "toggle the front object"
    elif action == Actions.done:
        return "do nothing"
    else:
        return RuntimeError(f"Invalid action {action}")


def crafter_action_to_str(action):
    from crafter import constants

    return constants.actions[action]


# def get_caption(env_type, descriptions, actions):
#     if env_type == "minigrid":
#         return minigrid_caption(descriptions, actions)
#     elif env_type == "crafter":
#         return crafter_caption(descriptions, actions)
#     else:
#         raise RuntimeError(f"Invalid env type {env_type} for captioning")


def get_caption(env_type, descriptions, actions):
    if env_type not in ["minigrid", "crafter"]:
        raise RuntimeError(f"Invalid env type {env_type} for captioning")
    caption_fn = minigrid_action_to_str if env_type == "minigrid" else crafter_action_to_str

    n_steps = len(actions)
    if n_steps > 1:
        desc_str = f"Agent history:\n"
        for i in range(n_steps):
            desc = (descriptions[i].strip() + ".") if not descriptions[i].strip().endswith(".") else descriptions[i].strip()
            desc_str += f"""Step {i + 1}: {desc} You choose the action: {caption_fn(actions[i])}.\n"""
    else:
        desc_str = f"""{descriptions[0]} You choose the action: {caption_fn(actions[0])}. """
    desc_str += f"Current observation: {descriptions[-1]}. \n"
    return desc_str


def get_caption_for_code(env_type, descriptions, actions):
    if env_type == "minigrid":
        n_steps = len(actions)
        desc_str = f"Agent history:\n"
        for i in range(n_steps):
            desc_str += f"""Step {i + 1}: {descriptions[i]} You choose the action: {minigrid_action_to_str(actions[i])}.\n"""
        desc_str += f"Current observation: {descriptions[-1]}. \n"
        return desc_str
    else:
        raise RuntimeError(f"Invalid env type {env_type} for captioning")


def minigrid_caption(descriptions, actions):
    n_steps = len(actions)

    desc_str = ""
    if n_steps != 1:
        for i in range(n_steps):
            desc_str += f"""Step {i + 1}: {descriptions[i]} You choose to {minigrid_action_to_str(actions[i])}.\n"""
    else:
        desc_str += f"""{descriptions[0]} You choose to {minigrid_action_to_str(actions[0])}.\n"""
    return desc_str


def crafter_caption(descriptions, actions):
    n_steps = len(descriptions)

    desc_str = ""
    if n_steps != 1:
        for i in range(n_steps):
            desc_str += f"""Step {i + 1}: {descriptions[i]} You choose to {crafter_action_to_str(actions[i])}. \n"""
    else:
        desc_str += f"""{descriptions[0]} You choose to {crafter_action_to_str(actions[0])}. \n"""
    return desc_str
