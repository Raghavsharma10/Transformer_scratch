def addLadder(settings):
    """define a new Ladder setting and save to disk file"""
    ladder = Ladder(settings)
    ladder.save()
    getKnownLadders()[ladder.name] = ladder
    return ladder