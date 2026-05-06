def playerSurrendered(cfg):
    """the player has forceibly left the game"""
    if cfg.numAgents + cfg.numBots == 2:
          otherResult = c.RESULT_VICTORY
    else: otherResult = c.RESULT_UNDECIDED # if multiple players remain, they need to finish the match
    return assignValue(cfg, c.RESULT_DEFEAT, otherResult)