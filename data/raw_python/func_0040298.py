def validate_filter(self, key, filter_value):
        """
        validate the filter key and value against the collection schema
        
        :param key: property name
        :param filter_value: value of the filter
        :returns True if all is ok otherwise False
        """
        ok = False
        seek = u"filter[%s]" % key
        value = None
        for link in self.schema['links']:
            if link['rel'] == 'instances':
               for property in link['properties']:
                   if seek == property:
                       value = link['properties'][property]
                       ok = True
        if not ok:
            return False
        ok = self._is_type(filter_value, value['type'])
        # if string with type add validation
        if ok is True and value['type'] == 'string' and 'format' in value.keys():
            ok = self._validate_json_format(filter_value, value)
            
        return ok