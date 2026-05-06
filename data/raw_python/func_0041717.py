def url(sport, league_id, team_id, start_date=None):
    """
    Given sport name, league_id, team_id, and optional start date (YYYY-MM-DD),
    return the url for the fantasy team page for that date (default: today)
    """
    url = '%s/%s/%s/' % (SPORT_URLS[sport], league_id, team_id)
    if start_date is not None:
        url += 'team?&date=%s' % start_date
    return url