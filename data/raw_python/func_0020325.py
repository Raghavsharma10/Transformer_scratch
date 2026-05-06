def Module(EPIC, campaign=None):
    '''
    Returns the module number for a given EPIC target.

    '''

    channel = Channel(EPIC, campaign=campaign)
    nums = {2: 1, 3: 5, 4: 9, 6: 13, 7: 17, 8: 21, 9: 25,
            10: 29, 11: 33, 12: 37, 13: 41, 14: 45, 15: 49,
            16: 53, 17: 57, 18: 61, 19: 65, 20: 69, 22: 73,
            23: 77, 24: 81}
    for c in [channel, channel - 1, channel - 2, channel - 3]:
        if c in nums.values():
            for mod, chan in nums.items():
                if chan == c:
                    return mod
    return None