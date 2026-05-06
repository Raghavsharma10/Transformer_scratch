def register_scheme(scheme, constructor):
        """
        Registers a scheme.

        :param str scheme: The scheme.
        :param callable constructor: The SimpleCondition constructor.
        """
        if not re.search(r'^[a-z]+$', scheme):
            raise ValueError('{0!s} is not a valid scheme'.format(scheme))

        if scheme in SimpleConditionFactory._constructors:
            raise ValueError('Scheme {0!s} is registered already'.format(scheme))

        SimpleConditionFactory._constructors[scheme] = constructor