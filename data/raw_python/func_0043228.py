def assignValue(cfg, playerValue, otherValue):
    """artificially determine match results given match circumstances.
    WARNING: cheating will be detected and your player will be banned from server"""
    player = cfg.whoAmI()
    result = {}
    for p in cfg.players:
        if p.name == player.name:   val = playerValue
        else:                       val = otherValue
        result[p.name] = val
    return result