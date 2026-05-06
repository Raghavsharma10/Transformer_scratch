def create_revlookup_query(self, *fulltext_searchterms, **keyvalue_searchterms):
        '''
        Create the part of the solr request that comes after the question mark,
        e.g. ?URL=*dkrz*&CHECKSUM=*abc*. If allowed search keys are
        configured, only these are used. If no'allowed search keys are
        specified, all key-value pairs are passed on to the reverse lookup
        servlet.

        :param fulltext_searchterms: Optional. Any term specified will be used
            as search term. Not implemented yet, so will be ignored.
        :param keyvalue_searchterms: Optional. Key-value pairs. Any key-value
            pair will be used to search for the value in the field "key".
            Wildcards accepted (refer to the documentation of the reverse
            lookup servlet for syntax.)
        :return: The query string, after the "?". If no valid search terms were
            specified, None is returned.
        '''
        LOGGER.debug('create_revlookup_query...')

        allowed_search_keys = self.__allowed_search_keys
        only_search_for_allowed_keys = False
        if len(allowed_search_keys) > 0:
            only_search_for_allowed_keys = True

        fulltext_searchterms_given = True
        fulltext_searchterms = b2handle.util.remove_value_none_from_list(fulltext_searchterms)
        if len(fulltext_searchterms) == 0:
            fulltext_searchterms_given = False
        
        if fulltext_searchterms_given:
            msg = 'Full-text search is not implemented yet.'+\
                ' The provided searchterms '+str(fulltext_searchterms)+\
                ' can not be used.'
            raise ReverseLookupException(msg=msg)

        keyvalue_searchterms_given = True
        keyvalue_searchterms = b2handle.util.remove_value_none_from_dict(keyvalue_searchterms)
        if len(keyvalue_searchterms) == 0:
            keyvalue_searchterms_given = False

        if not keyvalue_searchterms_given and not fulltext_searchterms_given:
            msg = 'No search terms have been specified. Please specify'+\
                ' at least one key-value-pair.'
            raise ReverseLookupException(msg=msg)

        counter = 0
        query = '?'
        for key, value in keyvalue_searchterms.items():

            if only_search_for_allowed_keys and key not in allowed_search_keys:
                msg = 'Cannot search for key "'+key+'". Only searches '+\
                    'for keys '+str(allowed_search_keys)+' are implemented.'
                raise ReverseLookupException(msg=msg)
            else:
                query = query+'&'+key+'='+value
                counter += 1

        query = query.replace('?&', '?')
        LOGGER.debug('create_revlookup_query: query: '+query)
        if counter == 0: # unreachable?
            msg = 'No valid search terms have been specified.'
            raise ReverseLookupException(msg=msg)
        return query