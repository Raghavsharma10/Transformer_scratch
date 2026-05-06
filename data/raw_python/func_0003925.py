def slice_match(sub, counter):
    """Efficiently test if counter is in ``xrange(*sub)``

       Arguments:
        | ``sub``  --  a slice object
        | ``counter``  -- an integer

       The function returns True if the counter is in
       ``xrange(sub.start, sub.stop, sub.step)``.
    """

    if sub.start is not None and counter < sub.start:
        return False
    if sub.stop is not None and counter >= sub.stop:
        raise StopIteration
    if sub.step is not None:
        if sub.start is None:
            if counter % sub.step != 0:
                return False
        else:
            if (counter - sub.start) % sub.step != 0:
                return False
    return True