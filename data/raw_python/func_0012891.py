def makeadistu_inlets(data, commdct):
    """make the dict adistu_inlets"""
    adistus = getadistus(data, commdct)
    # assume that the inlet node has the words "Air Inlet Node Name"
    airinletnode = "Air Inlet Node Name"
    adistu_inlets = {}
    for adistu in adistus:
        objkey = adistu.upper()
        objindex = data.dtls.index(objkey)
        objcomm = commdct[objindex]
        airinlets = []
        for i, comm in enumerate(objcomm):
            try:
                if comm['field'][0].find(airinletnode) != -1:
                    airinlets.append(comm['field'][0])
            except KeyError as err:
                pass
        adistu_inlets[adistu] = airinlets
    return adistu_inlets