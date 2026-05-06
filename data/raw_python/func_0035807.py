def from_api(cls, **kwargs):
        """Create a new instance from API arguments.

        This will switch camelCase keys into snake_case for instantiation.

        It will also identify any ``Instance`` or ``List`` properties, and
        instantiate the proper objects using the values. The end result being
        a fully Objectified and Pythonified API response.

        Returns:
            BaseModel: Instantiated model using the API values.
        """

        vals = cls.get_non_empty_vals({
            cls._to_snake_case(k): v for k, v in kwargs.items()
        })
        remove = []
        for attr, val in vals.items():
            try:
                vals[attr] = cls._parse_property(attr, val)
            except HelpScoutValidationException:
                remove.append(attr)
                logger.info(
                    'Unexpected property received in API response',
                    exc_info=True,
                )
        for attr in remove:
            del vals[attr]
        return cls(**cls.get_non_empty_vals(vals))