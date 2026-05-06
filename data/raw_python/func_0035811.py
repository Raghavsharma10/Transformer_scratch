def _parse_property(cls, name, value):
        """Parse a property received from the API into an internal object.

        Args:
            name (str): Name of the property on the object.
            value (mixed): The unparsed API value.

        Raises:
            HelpScoutValidationException: In the event that the property name
            is not found.

        Returns:
            mixed: A value compatible with the internal models.
        """

        prop = cls._props.get(name)
        return_value = value

        if not prop:
            logger.debug(
                '"%s" with value "%s" is not a valid property for "%s".' % (
                    name, value, cls,
                ),
            )
            return_value = None

        elif isinstance(prop, properties.Instance):
            return_value = prop.instance_class.from_api(**value)

        elif isinstance(prop, properties.List):
            return_value = cls._parse_property_list(prop, value)

        elif isinstance(prop, properties.Color):
            return_value = cls._parse_property_color(value)

        return return_value