def get_all_player_ids(ids="shots"):
    """
    Returns a pandas DataFrame containing the player IDs used in the
    stats.nba.com API.

    Parameters
    ----------
    ids : { "shots" | "all_players" | "all_data" }, optional
        Passing in "shots" returns a DataFrame that contains the player IDs of
        all players have shot chart data.  It is the default parameter value.

        Passing in "all_players" returns a DataFrame that contains
        all the player IDs used in the stats.nba.com API.

        Passing in "all_data" returns a DataFrame that contains all the data
        accessed from the JSON at the following url:
        http://stats.nba.com/stats/commonallplayers?IsOnlyCurrentSeason=0&LeagueID=00&Season=2015-16

        The column information for this DataFrame is as follows:
            PERSON_ID: The player ID for that player
            DISPLAY_LAST_COMMA_FIRST: The player's name.
            ROSTERSTATUS: 0 means player is not on a roster, 1 means he's on a
                          roster
            FROM_YEAR: The first year the player played.
            TO_YEAR: The last year the player played.
            PLAYERCODE: A code representing the player. Unsure of its use.

    Returns
    -------
    df : pandas DataFrame
        The pandas DataFrame object that contains the player IDs for the
        stats.nba.com API.

    """
    url = "http://stats.nba.com/stats/commonallplayers?IsOnlyCurrentSeason=0&LeagueID=00&Season=2015-16"

    # get the web page
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    # access 'resultSets', which is a list containing the dict with all the data
    # The 'header' key accesses the headers
    headers = response.json()['resultSets'][0]['headers']
    # The 'rowSet' key contains the player data along with their IDs
    players = response.json()['resultSets'][0]['rowSet']
    # Create dataframe with proper numeric types
    df = pd.DataFrame(players, columns=headers)

    # Dealing with different means of converision for pandas 0.17.0 or 0.17.1
    # and 0.15.0 or loweer
    if '0.17' in pd.__version__:
        # alternative to convert_objects() to numeric to get rid of warning
        # as convert_objects() is deprecated in pandas 0.17.0+
        df = df.apply(pd.to_numeric, args=('ignore',))
    else:
        df = df.convert_objects(convert_numeric=True)

    if ids == "shots":
        df = df.query("(FROM_YEAR >= 2001) or (TO_YEAR >= 2001)")
        df = df.reset_index(drop=True)
        # just keep the player ids and names
        df = df.iloc[:, 0:2]
        return df
    if ids == "all_players":
        df = df.iloc[:, 0:2]
        return df
    if ids == "all_data":
        return df
    else:
        er = "Invalid 'ids' value. It must be 'shots', 'all_shots', or 'all_data'."
        raise ValueError(er)