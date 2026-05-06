def __set_or_adapt_10320LOC_attributes(self, locelement, weight=None, http_role=None, **kvpairs):
        '''
        Adds or updates attributes of a <location> element. Existing attributes
            are not removed!

        :param locelement: A location element as xml snippet
            (xml.etree.ElementTree.Element).
        :param weight: Optional. The weight to be set (integer between 0 and
            1). If None, no weight attribute is set. If the value is outside
            the accepted range, it is set to 1.
        :param http_role: Optional. The http_role to be set. This accepts any
            string. Currently, Handle System can process 'conneg'. In future,
            it may be able to process 'no_conneg' and 'browser'.
        :param all others: Optional. All other key-value pairs will be set to
            the element. Any value is accepted and transformed to string.
        '''

        if weight is not None:
            LOGGER.debug('__set_or_adapt_10320LOC_attributes: weight (' + str(type(weight)) + '): ' + str(weight))
            weight = float(weight)
            if weight < 0  or weight > 1:
                default = 1
                LOGGER.debug('__set_or_adapt_10320LOC_attributes: Invalid weight (' + str(weight) + \
                    '), using default value (' + str(default) + ') instead.')
                weight = default
            weight = str(weight)
            locelement.set('weight', weight)

        if http_role is not None:
            locelement.set('http_role', http_role)

        for key, value in kvpairs.items():
            locelement.set(key, str(value))