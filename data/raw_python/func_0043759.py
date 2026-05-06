def _validate_dict(self, input_dict, schema_dict, path_to_root, object_title=''):

        ''' a helper method for recursively validating keys in dictionaries

        :return input_dict
        '''

    # reconstruct key path to current dictionary in model
        rules_top_level_key = re.sub('\[\d+\]', '[0]', path_to_root)
        map_rules = self.keyMap[rules_top_level_key]

    # construct list error report template
        map_error = {
            'object_title': object_title,
            'model_schema': self.schema,
            'input_criteria': map_rules,
            'failed_test': 'value_datatype',
            'input_path': path_to_root,
            'error_value': 0,
            'error_code': 4001
        }

    # validate map size
        if 'min_size' in map_rules.keys():
            input_size = sys.getsizeof(json.dumps(str(input_dict)).replace(' ','')) - 51
            if input_size < map_rules['min_size']:
                map_error['failed_test'] = 'min_size'
                map_error['error_value'] = input_size
                map_error['error_code'] = 4031
                raise InputValidationError(map_error)
        if 'max_size' in map_rules.keys():
            input_size = sys.getsizeof(json.dumps(str(input_dict)).replace(' ','')) - 51
            if input_size > map_rules['max_size']:
                map_error['failed_test'] = 'max_size'
                map_error['error_value'] = input_size
                map_error['error_code'] = 4032
                raise InputValidationError(map_error)

    # construct lists of keys in input dictionary
        input_keys = []
        input_key_list = []
        for key in input_dict.keys():
            error_dict = {
                'object_title': object_title,
                'model_schema': self.schema,
                'input_criteria': self.keyMap[rules_top_level_key],
                'failed_test': 'key_datatype',
                'input_path': path_to_root,
                'error_value': key,
                'error_code': 4004
            }
            error_dict['input_criteria']['key_datatype'] = 'string'
            if path_to_root == '.':
                if not isinstance(key, str):
                    input_key_name = path_to_root + str(key)
                    error_dict['input_path'] = input_key_name
                    raise InputValidationError(error_dict)
                input_key_name = path_to_root + key
            else:
                if not isinstance(key, str):
                    input_key_name = path_to_root + '.' + str(key)
                    error_dict['input_path'] = input_key_name
                    raise InputValidationError(error_dict)
                input_key_name = path_to_root + '.' + key
            input_keys.append(input_key_name)
            input_key_list.append(key)

    # TODO: validate top-level key and values against identical to reference

    # TODO: run lambda function and call validation

    # construct lists of keys in schema dictionary
        max_keys = []
        max_key_list = []
        req_keys = []
        req_key_list = []
        for key in schema_dict.keys():
            if path_to_root == '.':
                schema_key_name = path_to_root + key
            else:
                schema_key_name = path_to_root + '.' + key
            max_keys.append(schema_key_name)
            max_key_list.append(key)
            rules_schema_key_name = re.sub('\[\d+\]', '[0]', schema_key_name)
            if self.keyMap[rules_schema_key_name]['required_field']:
                req_keys.append(schema_key_name)
                req_key_list.append(key)

    # validate existence of required fields
        missing_keys = set(req_keys) - set(input_keys)
        if missing_keys:
            error_dict = {
                'object_title': object_title,
                'model_schema': self.schema,
                'input_criteria': self.keyMap[rules_top_level_key],
                'failed_test': 'required_field',
                'input_path': path_to_root,
                'error_value': list(missing_keys),
                'error_code': 4002
            }
            error_dict['input_criteria']['required_keys'] = req_keys
            raise InputValidationError(error_dict)

    # validate existence of extra fields
        extra_keys = set(input_keys) - set(max_keys)
        if extra_keys and not self.keyMap[rules_top_level_key]['extra_fields']:
            extra_key_list = []
            for key in extra_keys:
                pathless_key = re.sub(rules_top_level_key, '', key, count=1)
                extra_key_list.append(pathless_key)
            error_dict = {
                'object_title': object_title,
                'model_schema': self.schema,
                'input_criteria': self.keyMap[rules_top_level_key],
                'failed_test': 'extra_fields',
                'input_path': path_to_root,
                'error_value': extra_key_list,
                'error_code': 4003
            }
            error_dict['input_criteria']['maximum_scope'] = max_key_list
            raise InputValidationError(error_dict)

    # validate datatype of value
        for key, value in input_dict.items():
            if path_to_root == '.':
                input_key_name = path_to_root + key
            else:
                input_key_name = path_to_root + '.' + key
            rules_input_key_name = re.sub('\[\d+\]', '[0]', input_key_name)
            if input_key_name in max_keys:
                input_criteria = self.keyMap[rules_input_key_name]
                error_dict = {
                    'object_title': object_title,
                    'model_schema': self.schema,
                    'input_criteria': input_criteria,
                    'failed_test': 'value_datatype',
                    'input_path': input_key_name,
                    'error_value': value,
                    'error_code': 4001
                }
                try:
                    value_index = self._datatype_classes.index(value.__class__)
                except:
                    error_dict['error_value'] = value.__class__.__name__
                    raise InputValidationError(error_dict)
                value_type = self._datatype_names[value_index]
                if input_criteria['value_datatype'] == 'null':
                    pass
                else:
                    if value_type != input_criteria['value_datatype']:
                        raise InputValidationError(error_dict)

    # call appropriate validation sub-routine for datatype of value
                    if value_type == 'boolean':
                        input_dict[key] = self._validate_boolean(value, input_key_name, object_title)
                    elif value_type == 'number':
                        input_dict[key] = self._validate_number(value, input_key_name, object_title)
                    elif value_type == 'string':
                        input_dict[key] = self._validate_string(value, input_key_name, object_title)
                    elif value_type == 'map':
                        input_dict[key] = self._validate_dict(value, schema_dict[key], input_key_name, object_title)
                    elif value_type == 'list':
                        input_dict[key] = self._validate_list(value, schema_dict[key], input_key_name, object_title)

    # set default values for empty optional fields
        for key in max_key_list:
            if key not in input_key_list:
                indexed_key = max_keys[max_key_list.index(key)]
                if indexed_key in self.components.keys():
                    if 'default_value' in self.components[indexed_key]:
                        input_dict[key] = self.components[indexed_key]['default_value']

        return input_dict