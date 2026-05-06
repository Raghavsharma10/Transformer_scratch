def delLadder(name):
    """forget about a previously defined Ladder setting by deleting its disk file"""
    ladders = getKnownLadders()
    try:
        ladder = ladders[name]
        os.remove(ladder.filename) # delete from disk
        del ladders[name] # deallocate object
        return ladder
    except KeyError:
        raise ValueError("given ladder name '%s' is not a known ladder definition"%(name))