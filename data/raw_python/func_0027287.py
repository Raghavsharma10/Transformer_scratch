def parse(cls, value, default=_no_default):
        """Parses a flag integer or string into a Flags instance.

        Accepts the following types:
        - Members of this enum class. These are returned directly.
        - Integers. These are converted directly into a Flags instance with the given name.
        - Strings. The function accepts a comma-delimited list of flag names, corresponding to
          members of the enum. These are all ORed together.

        Examples:

        >>> class Car(Flags):
        ...     is_big = 1
        ...     has_wheels = 2
        >>> Car.parse(1)
        Car.is_big
        >>> Car.parse(3)
        Car.parse('has_wheels,is_big')
        >>> Car.parse('is_big,has_wheels')
        Car.parse('has_wheels,is_big')

        """
        if isinstance(value, cls):
            return value
        elif isinstance(value, int):
            e = cls._make_value(value)
        else:
            if not value:
                e = cls._make_value(0)
            else:
                r = 0
                for k in value.split(","):
                    v = cls._name_to_member.get(k, _no_default)
                    if v is _no_default:
                        if default is _no_default:
                            raise _create_invalid_value_error(cls, value)
                        else:
                            return default
                    r |= v.value
                e = cls._make_value(r)
        if not e.is_valid():
            if default is _no_default:
                raise _create_invalid_value_error(cls, value)
            return default
        return e