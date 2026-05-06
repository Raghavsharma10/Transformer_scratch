def select(yerrs, amps, amp_errs, widths):
    """ criteria for keeping an object """
    keep_1 = np.logical_and(amps < 0, widths > 1)
    keep_2 = np.logical_and(np.abs(amps) > 3*yerrs, amp_errs < 3*np.abs(amps))
    keep = np.logical_and(keep_1, keep_2)
    return keep