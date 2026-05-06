def summoner_names_to_id(summoners):
    """
    Gets a list of summoners names and return a dictionary mapping the player name to his/her summoner id
    :param summoners: a list of player names
    :return: a dictionary name -> id
    """
    ids = {}
    for start, end in _slice(0, len(summoners), 40):
        result = get_summoners_by_name(summoners[start:end])
        for name, summoner in result.items():
            ids[name] = summoner.id
    return ids