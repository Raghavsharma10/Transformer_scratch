def detect_mode(cls, **params):
        """Detect which listing mode of the given params.

        :params kwargs params: the params
        :return: one of the available modes
        :rtype: str
        :raises ValueError: if multiple modes are detected
        """
        modes = []
        for mode in cls.modes:
            if params.get(mode) is not None:
                modes.append(mode)
        if len(modes) > 1:
            error_message = 'ambiguous mode, must be one of {}'
            modes_csv = ', '.join(list(cls.modes))
            raise ValueError(error_message.format(modes_csv))
        return modes[0] if modes else cls.default_mode