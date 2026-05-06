def _mx(domain):
    """
    Return Deferred that fires with a list of (priority, MX domain) tuples for
    a given domain.
    """
    def got_records(result):
        return sorted(
            [(int(record.payload.preference), str(record.payload.name))
             for record in result[0]])
    d = lookupMailExchange(domain)
    d.addCallback(got_records)
    return d