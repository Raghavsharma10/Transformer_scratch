def get_all_team_ids():
    """Returns a pandas DataFrame with all Team IDs"""
    df = get_all_player_ids("all_data")
    df = pd.DataFrame({"TEAM_NAME": df.TEAM_NAME.unique(),
                       "TEAM_ID": df.TEAM_ID.unique()})
    return df