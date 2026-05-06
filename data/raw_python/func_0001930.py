def make_dnsentryitem_recordname(dns_name, condition='contains', negate=False, preserve_case=False):
    """
    Create a node for DnsEntryItem/RecordName
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'DnsEntryItem'
    search = 'DnsEntryItem/RecordName'
    content_type = 'string'
    content = dns_name
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node