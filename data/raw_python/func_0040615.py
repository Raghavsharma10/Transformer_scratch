def getKnownLadders(reset=False):
    """identify all of the currently defined ladders"""
    if not ladderCache or reset:
        jsonFiles = os.path.join(c.LADDER_FOLDER, "*.json")
        for ladderFilepath in glob.glob(jsonFiles):
            filename = os.path.basename(ladderFilepath)
            name = re.search("^ladder_(.*?).json$", filename).groups()[0]
            ladder = Ladder(name)
            ladderCache[ladder.name] = ladder
    return ladderCache