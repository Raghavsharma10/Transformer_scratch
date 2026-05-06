def addParts(parentPart, childPath, count, index):
    """
    BUILD A hierarchy BY REPEATEDLY CALLING self METHOD WITH VARIOUS childPaths
    count IS THE NUMBER FOUND FOR self PATH
    """
    if index == None:
        index = 0
    if index == len(childPath):
        return
    c = childPath[index]
    parentPart.count = coalesce(parentPart.count, 0) + count

    if parentPart.partitions == None:
        parentPart.partitions = FlatList()
    for i, part in enumerate(parentPart.partitions):
        if part.name == c.name:
            addParts(part, childPath, count, index + 1)
            return

    parentPart.partitions.append(c)
    addParts(c, childPath, count, index + 1)