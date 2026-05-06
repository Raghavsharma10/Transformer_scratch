def get_default_attribute_value(cls, object_class, property_name, attr_type=str):
        """ Gets the default value of a given property for a given object.

            These properties can be set in a config INI file looking like

            .. code-block:: ini

                [NUEntity]
                default_behavior = THIS
                speed = 1000

                [NUOtherEntity]
                attribute_name = a value

            This will be used when creating a :class:`bambou.NURESTObject` when no parameter or data is provided
        """

        if not cls._default_attribute_values_configuration_file_path:
            return None

        if not cls._config_parser:
            cls._read_config()

        class_name = object_class.__name__

        if not cls._config_parser.has_section(class_name):
            return None

        if not cls._config_parser.has_option(class_name, property_name):
            return None

        if sys.version_info < (3,):
            integer_types = (int, long,)
        else:
            integer_types = (int,)

        if isinstance(attr_type, integer_types):
            return cls._config_parser.getint(class_name, property_name)
        elif attr_type is bool:
            return cls._config_parser.getboolean(class_name, property_name)
        else:
            return cls._config_parser.get(class_name, property_name)