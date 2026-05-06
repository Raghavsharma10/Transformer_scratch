def getDomainNames(store):
    """
    Retrieve a list of all local domain names represented in the given store.
    """
    domains = set()
    domains.update(store.query(
            LoginMethod,
            AND(LoginMethod.internal == True,
                LoginMethod.domain != None)).getColumn("domain").distinct())
    return sorted(domains)