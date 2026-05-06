def cancelMatchRequest(cfg):
    """obtain information housed on the ladder about playerName"""
    payload = json.dumps([cfg.thePlayer])
    ladder = cfg.ladder
    return requests.post(
        url  = c.URL_BASE%(ladder.ipAddress, ladder.serverPort, "cancelmatch"),
        data = payload,
        #headers=headers,
    )