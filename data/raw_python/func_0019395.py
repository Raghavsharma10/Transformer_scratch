def get_team_id(team_name):
    """ Returns the team ID associated with the team name that is passed in.

    Parameters
    ----------
    team_name : str
        The team name whose ID we want.  NOTE: Only pass in the team name
        (e.g. "Lakers"), not the city, or city and team name, or the team
        abbreviation.

    Returns
    -------
    team_id : int
        The team ID associated with the team name.

    """
    df = get_all_team_ids()
    df = df[df.TEAM_NAME == team_name]
    if len(df) == 0:
        er = "Invalid team name or there is no team with that name."
        raise ValueError(er)
    team_id = df.TEAM_ID.iloc[0]
    return team_id