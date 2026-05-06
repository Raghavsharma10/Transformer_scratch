def get_one(table: LdapObjectClass, query: Optional[Q] = None,
            database: Optional[Database] = None, base_dn: Optional[str] = None) -> LdapObject:
    """ Get exactly one result from the database or fail. """
    results = search(table, query, database, base_dn)

    try:
        result = next(results)
    except StopIteration:
        raise ObjectDoesNotExist(f"Cannot find result for {query}.")

    try:
        next(results)
        raise MultipleObjectsReturned(f"Found multiple results for {query}.")
    except StopIteration:
        pass

    return result