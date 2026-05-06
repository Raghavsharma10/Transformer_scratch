def roll_group(group):
    """
    Rolls a group of dice in 2d6, 3d10, d12, etc. format

    :param group: String of dice group
    :return: Array of results
    """
    group = regex.match(r'^(\d*)d(\d+)$', group, regex.IGNORECASE)
    num_of_dice = int(group[1]) if group[1] != '' else 1
    type_of_dice = int(group[2])
    assert num_of_dice > 0

    result = []
    for i in range(num_of_dice):
        result.append(random.randint(1, type_of_dice))
    return result