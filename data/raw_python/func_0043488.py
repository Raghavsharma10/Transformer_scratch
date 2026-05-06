def ladderPlayerInfo(cfg, playerName, getMatchHistory=False):
    """obtain information housed on the ladder about playerName"""
    payload = json.dumps([playerName, getMatchHistory]) # if playerName == None, info on all players is retrieved
    ladder = cfg.ladder
    return requests.post(
        url  = c.URL_BASE%(ladder.ipAddress, ladder.serverPort, "playerstats"),
        data = payload,
        #headers=headers,
    )