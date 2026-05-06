def getidfobjectlist(idf):
    """return a list of all idfobjects in idf"""
    idfobjects = idf.idfobjects  
    # idfobjlst = [idfobjects[key] for key in idfobjects if idfobjects[key]]
    idfobjlst = [idfobjects[key] for key in idf.model.dtls if idfobjects[key]]
        # `for key in idf.model.dtls` maintains the order
        # `for key in idfobjects` does not have order
    idfobjlst = itertools.chain.from_iterable(idfobjlst)
    idfobjlst = list(idfobjlst)
    return idfobjlst