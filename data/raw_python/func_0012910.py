def getidfkeyswithnodes():
    """return a list of keys of idfobjects that hve 'None Name' fields"""
    idf = IDF(StringIO(""))
    keys = idfobjectkeys(idf)
    keysfieldnames = ((key, idf.newidfobject(key.upper()).fieldnames) 
                for key in keys)
    keysnodefdnames = ((key,  (name for name in fdnames 
                                if (name.endswith('Node_Name')))) 
                            for key, fdnames in keysfieldnames)
    nodekeys = [key for key, fdnames in keysnodefdnames if list(fdnames)]
    return nodekeys