def _build_fields(self):
        """ Builds a list of valid fields
        """
        declared_fields = self.solr._send_request('get', ADMIN_URL)
        result = decoder.decode(declared_fields)
        self.field_list = self._parse_fields(result, 'fields')

        # Build regular expressions to match dynamic fields.
        # dynamic field names may have exactly one wildcard, either at
        # the beginning or the end of the name
        self._dynamic_field_regexes = []
        for wc_pattern in self._parse_fields(result, 'dynamicFields'):
            if wc_pattern[0] == "*":
                self._dynamic_field_regexes.append(
                    re.compile(".*%s\Z" % wc_pattern[1:]))
            elif wc_pattern[-1] == "*":
                self._dynamic_field_regexes.append(
                    re.compile("\A%s.*" % wc_pattern[:-1]))