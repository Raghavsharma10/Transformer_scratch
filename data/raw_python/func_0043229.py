def idPlayerResults(cfg, rawResult):
    """interpret standard rawResult for all players with known IDs"""
    result = {}
    knownPlayers = []
    dictResult = {plyrRes.player_id : plyrRes.result for plyrRes in rawResult}
    for p in cfg.players:
        if p.playerID and p.playerID in dictResult: # identified player w/ result
            knownPlayers.append(p)
            result[p.name] = dictResult[p.playerID]
    #if len(knownPlayers) == len(dictResult) - 1: # identified all but one player
    #    for p in cfg.players: # search for the not identified player
    #        if p in knownPlayers: continue # already found
    #        result.append( [p.name, p.playerID, dictResult[p.playerID]] )
    #        break # found missing player; stop searching
    #for r in result:
    #    print("result:>", r)
    return result