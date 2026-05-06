def get_property(self, regex):
        """Gets the name of a specific property

        This public method is passed a regular expression and
        returns the matching property name. If either the property
        is not found or if the passed string matches more than one
        property, this function will return None.

        :param regex: Regular expression to search on
        :return: (str) Property name matching the passed regex or None.
        """
        log = logging.getLogger(self.cls_logger + '.get_property')

        if not isinstance(regex, basestring):
            log.error('regex arg is not a string found type: {t}'.format(t=regex.__class__.__name__))
            return None

        log.debug('Looking up property based on regex: {r}'.format(r=regex))
        prop_list_matched = []
        for prop_name in self.properties.keys():
            match = re.search(regex, prop_name)
            if match:
                prop_list_matched.append(prop_name)
        if len(prop_list_matched) == 1:
            log.debug('Found matching property: {p}'.format(p=prop_list_matched[0]))
            return prop_list_matched[0]
        elif len(prop_list_matched) > 1:
            log.debug('Passed regex {r} matched more than 1 property, checking for an exact match...'.format(r=regex))
            for matched_prop in prop_list_matched:
                if matched_prop == regex:
                    log.debug('Found an exact match: {p}'.format(p=matched_prop))
                    return matched_prop
            log.debug('Exact match not found for regex {r}, returning None'.format(r=regex))
            return None
        else:
            log.debug('Passed regex did not match any deployment properties: {r}'.format(r=regex))
            return None