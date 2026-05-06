def _parse_fields(self, result, field_name):
        """ If Schema access, parse fields and build respective lists
        """
        field_list = []
        for key, value in result.get('schema', {}).get(field_name, {}).items():
            if key not in field_list:
                field_list.append(key)
        return field_list