def get_player_id(player):
    """
    Returns the player ID(s) associated with the player name that is passed in.

    There are instances where players have the same name so there are multiple
    player IDs associated with it.

    Parameters
    ----------
    player : str
        The desired player's name in 'Last Name, First Name' format. Passing in
        a single name returns a numpy array containing all the player IDs
        associated with that name.

    Returns
    -------
    player_id : numpy array
        The numpy array that contains the player ID(s).

    """
    players_df = get_all_player_ids("all_data")
    player = players_df[players_df.DISPLAY_LAST_COMMA_FIRST == player]
    # if there are no plyaers by the given name, raise an a error
    if len(player) == 0:
        er = "Invalid player name passed or there is no player with that name."
        raise ValueError(er)
    player_id = player.PERSON_ID.values
    return player_id