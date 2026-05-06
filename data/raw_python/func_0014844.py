def anim(host, seq, anim, d):
    """
    Makes the drone execute a predefined movement (animation).

    Parameters:
    seq -- sequcence number
    anim -- Integer: animation to play
    d -- Integer: total duration in seconds of the animation
    """
    at(host, 'ANIM', seq, [anim, d])