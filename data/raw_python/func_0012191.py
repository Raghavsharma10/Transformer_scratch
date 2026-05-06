def reducejson(j):
    """ 
    Not sure if there's a better way to walk the ... interesting result
    """

    authors = []

    for key in j["data"]["repository"]["commitComments"]["edges"]:
            authors.append(key["node"]["author"])

    for key in j["data"]["repository"]["issues"]["nodes"]:
            authors.append(key["author"])
            for c in key["comments"]["nodes"]:
                    authors.append(c["author"])
            
    for key in j["data"]["repository"]["pullRequests"]["edges"]:
            authors.append(key["node"]["author"])
            for c in key["node"]["comments"]["nodes"]:
                    authors.append(c["author"])

    unique = list({v['login']:v for v in authors if v is not None}.values())
    return unique