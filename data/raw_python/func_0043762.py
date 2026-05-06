def _validate_string(self, input_string, path_to_root, object_title=''):

        '''
            a helper method for validating properties of a string

        :return: input_string
        '''

        rules_path_to_root = re.sub('\[\d+\]', '[0]', path_to_root)
        input_criteria = self.keyMap[rules_path_to_root]
        error_dict = {
            'object_title': object_title,
            'model_schema': self.schema,
            'input_criteria': input_criteria,
            'failed_test': 'value_datatype',
            'input_path': path_to_root,
            'error_value': input_string,
            'error_code': 4001
        }
        if 'byte_data' in input_criteria.keys():
            if input_criteria['byte_data']:
                error_dict['failed_test'] = 'byte_data'
                error_dict['error_code'] = 4011
                try:
                    decoded_bytes = b64decode(input_string)
                except:
                    raise InputValidationError(error_dict)
                if not isinstance(decoded_bytes, bytes):
                    raise InputValidationError(error_dict)
        if 'min_value' in input_criteria.keys():
            if input_string < input_criteria['min_value']:
                error_dict['failed_test'] = 'min_value'
                error_dict['error_code'] = 4022
                raise InputValidationError(error_dict)
        if 'max_value' in input_criteria.keys():
            if input_string > input_criteria['max_value']:
                error_dict['failed_test'] = 'max_value'
                error_dict['error_code'] = 4023
                raise InputValidationError(error_dict)
        if 'greater_than' in input_criteria.keys():
            if input_string <= input_criteria['greater_than']:
                error_dict['failed_test'] = 'greater_than'
                error_dict['error_code'] = 4024
                raise InputValidationError(error_dict)
        if 'less_than' in input_criteria.keys():
            if input_string >= input_criteria['less_than']:
                error_dict['failed_test'] = 'less_than'
                error_dict['error_code'] = 4025
                raise InputValidationError(error_dict)
        if 'equal_to' in input_criteria.keys():
            if input_string != input_criteria['equal_to']:
                error_dict['failed_test'] = 'equal_to'
                error_dict['error_code'] = 4026
                raise InputValidationError(error_dict)
        if 'min_length' in input_criteria.keys():
            if len(input_string) < input_criteria['min_length']:
                error_dict['failed_test'] = 'min_length'
                error_dict['error_code'] = 4012
                raise InputValidationError(error_dict)
        if 'max_length' in input_criteria.keys():
            if len(input_string) > input_criteria['max_length']:
                error_dict['failed_test'] = 'max_length'
                error_dict['error_code'] = 4013
                raise InputValidationError(error_dict)
        if 'must_not_contain' in input_criteria.keys():
            for regex in input_criteria['must_not_contain']:
                regex_pattern = re.compile(regex)
                if regex_pattern.findall(input_string):
                    error_dict['failed_test'] = 'must_not_contain'
                    error_dict['error_code'] = 4014
                    raise InputValidationError(error_dict)
        if 'must_contain' in input_criteria.keys():
            for regex in input_criteria['must_contain']:
                regex_pattern = re.compile(regex)
                if not regex_pattern.findall(input_string):
                    error_dict['failed_test'] = 'must_contain'
                    error_dict['error_code'] = 4015
                    raise InputValidationError(error_dict)
        if 'contains_either' in input_criteria.keys():
            regex_match = False
            for regex in input_criteria['contains_either']:
                regex_pattern = re.compile(regex)
                if regex_pattern.findall(input_string):
                    regex_match = True
            if not regex_match:
                error_dict['failed_test'] = 'contains_either'
                error_dict['error_code'] = 4016
                raise InputValidationError(error_dict)
        if 'discrete_values' in input_criteria.keys():
            if input_string not in input_criteria['discrete_values']:
                error_dict['failed_test'] = 'discrete_values'
                error_dict['error_code'] = 4041
                raise InputValidationError(error_dict)
        if 'excluded_values' in input_criteria.keys():
            if input_string in input_criteria['excluded_values']:
                error_dict['failed_test'] = 'excluded_values'
                error_dict['error_code'] = 4042
                raise InputValidationError(error_dict)

    # TODO: validate string against identical to reference

    # TODO: run lambda function and call validation url

        return input_string