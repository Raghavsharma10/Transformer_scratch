def uniqueTags(tagList):
    '''
        uniqueTags - Returns the unique tags in tagList.
        
            @param tagList list<AdvancedTag> : A list of tag objects.
    '''
    ret = []
    alreadyAdded = set()
    for tag in tagList:
        myUid = tag.getUid()
        if myUid in alreadyAdded:
            continue
        ret.append(tag)
    return TagCollection(ret)