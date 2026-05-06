def parse(cls, value, default=_no_default):
        """Parses an enum member name or value into an enum member.

        Accepts the following types:
        - Members of this enum class. These are returned directly.
        - Integers. If there is an enum member with the integer as a value, that member is returned.
        - Strings. If there is an enum member with the string as its name, that member is returned.
        For integers and strings that don't correspond to an enum member, default is returned; if
        no default is given the function raises KeyError instead.

        Examples:

        >>> class Color(Enum):
        ...     red = 1
        ...     blue = 2
        >>> Color.parse(Color.red)
        Color.red
        >>> Color.parse(1)
        Color.red
        >>> Color.parse('blue')
        Color.blue

        """
        if isinstance(value, cls):
            return value
        elif isinstance(value, six.integer_types) and not isinstance(value, EnumBase):
            e = cls._value_to_member.get(value, _no_default)
        else:
            e = cls._name_to_member.get(value, _no_default)
        if e is _no_default or not e.is_valid():
            if default is _no_default:
                raise _create_invalid_value_error(cls, value)
            return default
        return e