def ref(host, seq, takeoff, emergency=False):
    """
    Basic behaviour of the drone: take-off/landing, emergency stop/reset)

    Parameters:
    seq -- sequence number
    takeoff -- True: Takeoff / False: Land
    emergency -- True: Turn off the engines
    """
    p = 0b10001010101000000000000000000
    if takeoff:
        p |= 0b1000000000
    if emergency:
        p |= 0b100000000
    at(host, 'REF', seq, [p])