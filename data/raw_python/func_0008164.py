def paginate(query, org_id, max_pages=maxsize, max_records=maxsize):
    """
    Paginate through all results of a UMAPI query
    :param query: a query method from a UMAPI instance (callable as a function)
    :param org_id: the organization being queried
    :param max_pages: the max number of pages to collect before returning (default all)
    :param max_records: the max number of records to collect before returning (default all)
    :return: the queried records
    """
    page_count = 0
    record_count = 0
    records = []
    while page_count < max_pages and record_count < max_records:
        res = make_call(query, org_id, page_count)
        page_count += 1
        # the following incredibly ugly piece of code is very fragile.
        # the problem is that we are a "dumb helper" that doesn't understand
        # the semantics of the UMAPI or know which query we were given.
        if "groups" in res:
            records += res["groups"]
        elif "users" in res:
            records += res["users"]
        record_count = len(records)
        if res.get("lastPage"):
            break
    return records