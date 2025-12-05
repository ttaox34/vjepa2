from typing import Any, Tuple


def unpack_sample(
    sample: Tuple[Any, ...],
    include_returns: bool = False,
    include_action_latents: bool = False,
    include_raw_clips: bool = False,
):
    """
    Standardized tuple unpacker for RetroGameDataset samples.

    Returns:
        clips, actions, states, extrinsics, rewards, returns, action_latents, raw_clips, indices
        (some entries may be None depending on flags)
    """

    cursor = 0
    clips = sample[cursor]
    cursor += 1
    actions = sample[cursor]
    cursor += 1
    states = sample[cursor]
    cursor += 1
    extrinsics = sample[cursor]
    cursor += 1
    rewards = sample[cursor]
    cursor += 1

    returns = None
    if include_returns:
        returns = sample[cursor]
        cursor += 1

    action_latents = None
    if include_action_latents:
        action_latents = sample[cursor]
        cursor += 1

    raw_clips = None
    if include_raw_clips:
        raw_clips = sample[cursor]
        cursor += 1

    indices = sample[cursor]
    return clips, actions, states, extrinsics, rewards, returns, action_latents, raw_clips, indices
