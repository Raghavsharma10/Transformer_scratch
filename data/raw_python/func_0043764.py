def _ingest_dict(self, input_dict, schema_dict, path_to_root):

        '''
            a helper method for ingesting keys, value pairs in a dictionary

        :return: valid_dict
        '''

        valid_dict = {}

    # construct path to root for rules
        rules_path_to_root = re.sub('\[\d+\]', '[0]', path_to_root)

    # iterate over keys in schema dict
        for key, value in schema_dict.items():
            key_path = path_to_root
            if not key_path == '.':
                key_path += '.'
            key_path += key
            rules_key_path = re.sub('\[\d+\]', '[0]', key_path)
            value_match = False
            if key in input_dict.keys():
                value_index = self._datatype_classes.index(value.__class__)
                value_type = self._datatype_names[value_index]
                try:
                    v_index = self._datatype_classes.index(input_dict[key].__class__)
                    v_type = self._datatype_names[v_index]
                    if v_type == value_type:
                        value_match = True
                except:
                    value_match = False
            if value_match:
                if value_type == 'null':
                    valid_dict[key] = input_dict[key]
                elif value_type == 'boolean':
                    valid_dict[key] = self._ingest_boolean(input_dict[key], key_path)
                elif value_type == 'number':
                    valid_dict[key] = self._ingest_number(input_dict[key], key_path)
                elif value_type == 'string':
                    valid_dict[key] = self._ingest_string(input_dict[key], key_path)
                elif value_type == 'map':
                    valid_dict[key] = self._ingest_dict(input_dict[key], schema_dict[key], key_path)
                elif value_type == 'list':
                    valid_dict[key] = self._ingest_list(input_dict[key], schema_dict[key], key_path)
            else:
                value_type = self.keyMap[rules_key_path]['value_datatype']
                if 'default_value' in self.keyMap[rules_key_path]:
                    valid_dict[key] = self.keyMap[rules_key_path]['default_value']
                elif value_type == 'null':
                    valid_dict[key] = None
                elif value_type == 'boolean':
                    valid_dict[key] = False
                elif value_type == 'number':
                    valid_dict[key] = 0.0
                    if 'integer_data' in self.keyMap[rules_key_path].keys():
                        if self.keyMap[rules_key_path]['integer_data']:
                            valid_dict[key] = 0
                elif value_type == 'string':
                    valid_dict[key] = ''
                elif value_type == 'list':
                    valid_dict[key] = []
                elif value_type == 'map':
                    valid_dict[key] = self._ingest_dict({}, schema_dict[key], key_path)

    # add extra fields if set to True
        if self.keyMap[rules_path_to_root]['extra_fields']:
            for key, value in input_dict.items():
                if key not in valid_dict.keys():
                    valid_dict[key] = value

        return valid_dict