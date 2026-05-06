def validate(self):
        """ Validate the current object attributes.

            Check all attributes and store errors

            Returns:
                Returns True if all attibutes of the object
                respect contraints. Returns False otherwise and
                store error in errors dict.

        """
        self._attribute_errors = dict()  # Reset validation errors

        for local_name, attribute in self._attributes.items():

            value = getattr(self, local_name, None)

            if attribute.is_required and (value is None or value == ""):
                self._attribute_errors[local_name] = {'title': 'Invalid input',
                                                      'description': 'This value is mandatory.',
                                                      'remote_name': attribute.remote_name}
                continue

            if value is None:
                continue  # without error

            if not self._validate_type(local_name, attribute.remote_name, value, attribute.attribute_type):
                continue

            if attribute.min_length is not None and len(value) < attribute.min_length:
                self._attribute_errors[local_name] = {'title': 'Invalid length',
                                                      'description': 'Attribute %s minimum length should be %s but is %s' % (attribute.remote_name, attribute.min_length, len(value)),
                                                      'remote_name': attribute.remote_name}
                continue

            if attribute.max_length is not None and len(value) > attribute.max_length:
                self._attribute_errors[local_name] = {'title': 'Invalid length',
                                                      'description': 'Attribute %s maximum length should be %s but is %s' % (attribute.remote_name, attribute.max_length, len(value)),
                                                      'remote_name': attribute.remote_name}
                continue

            if attribute.attribute_type == list:
                valid = True
                for item in value:
                    if valid is True:
                        valid = self._validate_value(local_name, attribute, item)
            else:
                self._validate_value(local_name, attribute, value)

        return self.is_valid()