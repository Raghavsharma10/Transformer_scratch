def _validate_number(self, input_number, path_to_root, object_title=''):

        '''
            a helper method for validating properties of a number

        :return: input_number
        '''

        rules_path_to_root = re.sub('\[\d+\]', '[0]', path_to_root)
        input_criteria = self.keyMap[rules_path_to_root]
        error_dict = {
            'object_title': object_title,
            'model_schema': self.schema,
            'input_criteria': input_criteria,
            'failed_test': 'value_datatype',
            'input_path': path_to_root,
            'error_value': input_number,
            'error_code': 4001
        }
        if 'integer_data' in input_criteria.keys():
            if input_criteria['integer_data'] and not isinstance(input_number, int):
                error_dict['failed_test'] = 'integer_data'
                error_dict['error_code'] = 4021
                raise InputValidationError(error_dict)
        if 'min_value' in input_criteria.keys():
            if input_number < input_criteria['min_value']:
                error_dict['failed_test'] = 'min_value'
                error_dict['error_code'] = 4022
                raise InputValidationError(error_dict)
        if 'max_value' in input_criteria.keys():
            if input_number > input_criteria['max_value']:
                error_dict['failed_test'] = 'max_value'
                error_dict['error_code'] = 4023
                raise InputValidationError(error_dict)
        if 'greater_than' in input_criteria.keys():
            if input_number <= input_criteria['greater_than']:
                error_dict['failed_test'] = 'greater_than'
                error_dict['error_code'] = 4024
                raise InputValidationError(error_dict)
        if 'less_than' in input_criteria.keys():
            if input_number >= input_criteria['less_than']:
                error_dict['failed_test'] = 'less_than'
                error_dict['error_code'] = 4025
                raise InputValidationError(error_dict)
        if 'equal_to' in input_criteria.keys():
            if input_number != input_criteria['equal_to']:
                error_dict['failed_test'] = 'equal_to'
                error_dict['error_code'] = 4026
                raise InputValidationError(error_dict)
        if 'discrete_values' in input_criteria.keys():
            if input_number not in input_criteria['discrete_values']:
                error_dict['failed_test'] = 'discrete_values'
                error_dict['error_code'] = 4041
                raise InputValidationError(error_dict)
        if 'excluded_values' in input_criteria.keys():
            if input_number in input_criteria['excluded_values']:
                error_dict['failed_test'] = 'excluded_values'
                error_dict['error_code'] = 4042
                raise InputValidationError(error_dict)

    # TODO: validate number against identical to reference

    # TODO: run lambda function and call validation url

        return input_number