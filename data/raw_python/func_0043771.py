def validate(self, input_data, path_to_root='', object_title=''):

        '''
            a core method for validating input against the model

            input_data is only returned if all data is valid

        :param input_data: list, dict, string, number, or boolean to validate
        :param path_to_root: [optional] string with dot-path of model component
        :param object_title: [optional] string with name of input to validate
        :return: input_data (or InputValidationError)
        '''

        __name__ = '%s.validate' % self.__class__.__name__
        _path_arg = '%s(path_to_root="...")' % __name__
        _title_arg = '%s(object_title="...")' % __name__

    # validate input
        copy_path = path_to_root
        if path_to_root:
            if not isinstance(path_to_root, str):
                raise ModelValidationError('%s must be a string.' % _path_arg)
            else:
                if path_to_root[0] != '.':
                    copy_path = '.%s' % path_to_root
                if not copy_path in self.keyMap.keys():
                    raise ModelValidationError('%s does not exist in components %s.' % (_path_arg.replace('...', path_to_root), self.keyMap.keys()))
        else:
            copy_path = '.'
        if object_title:
            if not isinstance(object_title, str):
                raise ModelValidationError('%s must be a string' % _title_arg)

    # construct generic error dictionary
        error_dict = {
            'object_title': object_title,
            'model_schema': self.schema,
            'input_criteria': self.keyMap[copy_path],
            'failed_test': 'value_datatype',
            'input_path': copy_path,
            'error_value': input_data,
            'error_code': 4001
        }

    # determine value type of input data
        try:
            input_index = self._datatype_classes.index(input_data.__class__)
        except:
            error_dict['error_value'] = input_data.__class__.__name__
            raise InputValidationError(error_dict)
        input_type = self._datatype_names[input_index]

    # validate input data type
        if input_type != self.keyMap[copy_path]['value_datatype']:
            raise InputValidationError(error_dict)

    # run helper method appropriate to data type
        if input_type == 'boolean':
            input_data = self._validate_boolean(input_data, copy_path, object_title)
        elif input_type == 'number':
            input_data = self._validate_number(input_data, copy_path, object_title)
        elif input_type == 'string':
            input_data = self._validate_string(input_data, copy_path, object_title)
        elif input_type == 'list':
            schema_list = self._reconstruct(copy_path)
            input_data = self._validate_list(input_data, schema_list, copy_path, object_title)
        elif input_type == 'map':
            schema_dict = self._reconstruct(copy_path)
            input_data = self._validate_dict(input_data, schema_dict, copy_path, object_title)

        return input_data