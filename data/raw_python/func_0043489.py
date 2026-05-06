def reportMatchCompletion(cfg, results, replayData):
    """send information back to the server about the match's winners/losers"""
    payload = json.dumps([cfg.flatten(), results, replayData])
    ladder = cfg.ladder
    return requests.post(
        url  = c.URL_BASE%(ladder.ipAddress, ladder.serverPort, "matchfinished"),
        data = payload,
        #headers=headers,
    )