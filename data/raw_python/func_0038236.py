def _create_instance_attributes(self, arguments):
        """
        Copies class level attribute templates and makes instance placeholders

        This step is required for direct uses of Model classes. This creates a
        copy of attribute_names ignores methods and private variables.
        DataCollection types are deep copied to ignore memory reference conflicts.

        DataType instances are initialized to None or default value.
        """
        for attribute_name, type_instance in self.getmembers():
            if isinstance(type_instance, DataType):
                self._templates[attribute_name] = type_instance

                value = None
                if attribute_name in arguments:
                    value = arguments[attribute_name]

                try:
                    self._attributes[attribute_name] = type_instance.validate(value)
                # we can safely ignore required warnings during initialization
                except exception.RequiredAttributeError:
                    self._attributes[attribute_name] = None