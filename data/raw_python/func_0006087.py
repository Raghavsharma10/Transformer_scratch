def is_subdomain(domain, reference):
    """Tests if a hostname is a subdomain of a reference hostname
    e.g. www.domain.com is subdomain of reference

    @param domain: Domain to test if it is a subdomain
    @param reference: Reference "parent" domain
    """
    index_of_reference = domain.find(reference)
    if index_of_reference > 0 and domain[index_of_reference:] == reference:
        return True
    return False