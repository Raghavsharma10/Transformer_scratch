def search(self, base, scope, filterstr='(objectClass=*)',
               attrlist=None, limit=None) -> Generator[Tuple[str, dict], None, None]:
        """
        Search for entries in LDAP database.
        """

        _debug("search", base, scope, filterstr, attrlist, limit)

        # first results
        if attrlist is None:
            attrlist = ldap3.ALL_ATTRIBUTES
        elif isinstance(attrlist, set):
            attrlist = list(attrlist)

        def first_results(obj):
            _debug("---> searching ldap", limit)
            obj.search(
                base, filterstr, scope, attributes=attrlist, paged_size=limit)
            return obj.response

        # get the 1st result
        result_list = self._do_with_retry(first_results)

        # Loop over list of search results
        for result_item in result_list:
            # skip searchResRef for now
            if result_item['type'] != "searchResEntry":
                continue
            dn = result_item['dn']
            attributes = result_item['raw_attributes']
            # did we already retrieve this from cache?
            _debug("---> got ldap result", dn)
            _debug("---> yielding", result_item)
            yield (dn, attributes)

        # we are finished - return results, eat cake
        _debug("---> done")
        return