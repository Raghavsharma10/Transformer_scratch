def Channels(module):
    '''
    Returns the channels contained in the given K2 module.

    '''

    nums = {2: 1, 3: 5, 4: 9, 6: 13, 7: 17, 8: 21, 9: 25,
            10: 29, 11: 33, 12: 37, 13: 41, 14: 45, 15: 49,
            16: 53, 17: 57, 18: 61, 19: 65, 20: 69, 22: 73,
            23: 77, 24: 81}

    if module in nums:
        return [nums[module], nums[module] + 1,
                nums[module] + 2, nums[module] + 3]
    else:
        return None