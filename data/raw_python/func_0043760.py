def _validate_list(self, input_list, schema_list, path_to_root, object_title=''):

        '''
            a helper method for recursively validating items in a list

        :return: input_list
        '''

    # construct rules for list and items
        rules_path_to_root = re.sub('\[\d+\]', '[0]', path_to_root)
        list_rules = self.keyMap[rules_path_to_root]
        initial_key = rules_path_to_root + '[0]'
        item_rules = self.keyMap[initial_key]

    # construct list error report template
        list_error = {
            'object_title': object_title,
            'model_schema': self.schema,
            'input_criteria': list_rules,
            'failed_test': 'value_datatype',
            'input_path': path_to_root,
            'error_value': 0,
            'error_code': 4001
        }

    # validate list rules
        if 'min_size' in list_rules.keys():
            if len(input_list) < list_rules['min_size']:
                list_error['failed_test'] = 'min_size'
                list_error['error_value'] = len(input_list)
                list_error['error_code'] = 4031
                raise InputValidationError(list_error)
        if 'max_size' in list_rules.keys():
            if len(input_list) > list_rules['max_size']:
                list_error['failed_test'] = 'max_size'
                list_error['error_value'] = len(input_list)
                list_error['error_code'] = 4032
                raise InputValidationError(list_error)

    # construct item error report template
        item_error = {
            'object_title': object_title,
            'model_schema': self.schema,
            'input_criteria': item_rules,
            'failed_test': 'value_datatype',
            'input_path': initial_key,
            'error_value': None,
            'error_code': 4001
        }

    # validate datatype of items
        for i in range(len(input_list)):
            input_path = path_to_root + '[%s]' % i
            item = input_list[i]
            item_error['input_path'] = input_path
            try:
                item_index = self._datatype_classes.index(item.__class__)
            except:
                item_error['error_value'] = item.__class__.__name__
                raise InputValidationError(item_error)
            item_type = self._datatype_names[item_index]
            item_error['error_value'] = item
            if item_rules['value_datatype'] == 'null':
                pass
            else:
                if item_type != item_rules['value_datatype']:
                    raise InputValidationError(item_error)

    # call appropriate validation sub-routine for datatype of item
                if item_type == 'boolean':
                    input_list[i] = self._validate_boolean(item, input_path, object_title)
                elif item_type == 'number':
                    input_list[i] = self._validate_number(item, input_path, object_title)
                elif item_type == 'string':
                    input_list[i] = self._validate_string(item, input_path, object_title)
                elif item_type == 'map':
                    input_list[i] = self._validate_dict(item, schema_list[0], input_path, object_title)
                elif item_type == 'list':
                    input_list[i] = self._validate_list(item, schema_list[0], input_path, object_title)

    # validate unique values in list
        if 'unique_values' in list_rules.keys():
            if len(set(input_list)) < len(input_list):
                list_error['failed_test'] = 'unique_values'
                list_error['error_value'] = input_list
                list_error['error_code'] = 4033
                raise InputValidationError(list_error)

    # TODO: validate top-level item values against identical to reference

    # TODO: run lambda function and call validation url

        return input_list