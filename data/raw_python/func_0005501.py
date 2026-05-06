def get_matching_property_names(self, regex):
        """Returns a list of property names matching the provided
        regular expression

        :param regex: Regular expression to search on
        :return: (list) of property names matching the regex
        """
        log = logging.getLogger(self.cls_logger + '.get_matching_property_names')
        prop_list_matched = []
        if not isinstance(regex, basestring):
            log.warn('regex arg is not a string, found type: {t}'.format(t=regex.__class__.__name__))
            return prop_list_matched
        log.debug('Finding properties matching regex: {r}'.format(r=regex))
        for prop_name in self.properties.keys():
            match = re.search(regex, prop_name)
            if match:
                prop_list_matched.append(prop_name)
        return prop_list_matched