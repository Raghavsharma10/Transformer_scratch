def pwm(host, seq, m1, m2, m3, m4):
    """
    Sends control values directly to the engines, overriding control loops.

    Parameters:
    seq -- sequence number
    m1 -- Integer: front left command
    m2 -- Integer: front right command
    m3 -- Integer: back right command
    m4 -- Integer: back left command
    """
    at(host, 'PWM', seq, [m1, m2, m3, m4])