def _validate_boolean(self, input_boolean, path_to_root, object_title=''):

        '''
            a helper method for validating properties of a boolean

        :return: input_boolean
        '''

        rules_path_to_root = re.sub('\[\d+\]', '[0]', path_to_root)
        input_criteria = self.keyMap[rules_path_to_root]
        error_dict = {
            'object_title': object_title,
            'model_schema': self.schema,
            'input_criteria': input_criteria,
            'failed_test': 'value_datatype',
            'input_path': path_to_root,
            'error_value': input_boolean,
            'error_code': 4001
        }
        if 'equal_to' in input_criteria.keys():
            if input_boolean != input_criteria['equal_to']:
                error_dict['failed_test'] = 'equal_to'
                error_dict['error_code'] = 4026
                raise InputValidationError(error_dict)

    # TODO: validate boolean against identical to reference

    # TODO: run lambda function and call validation url

        return input_boolean