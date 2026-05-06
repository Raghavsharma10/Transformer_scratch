def led(host, seq, anim, f, d):
    """
    Control the drones LED.

    Parameters:
    seq -- sequence number
    anim -- Integer: animation to play
    f -- Float: frequency in HZ of the animation
    d -- Integer: total duration in seconds of the animation
    """
    at(host, 'LED', seq, [anim, float(f), d])