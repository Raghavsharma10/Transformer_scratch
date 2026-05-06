def _get_converter(self, converter_str):
        """find converter function reference by name

        find converter by name, converter name follows this convention:

            Class.method

        or:

            method

        The first type of converter class/function must be available in
        current module.
        The second type of converter must be available in `__builtin__`
        (or `builtins` in python3) module.

        :param converter_str: string representation of the converter func
        :return: function reference
        """
        ret = None
        if converter_str is not None:
            converter_desc_list = converter_str.split('.')
            if len(converter_desc_list) == 1:
                converter = converter_desc_list[0]
                # default to `converter`
                ret = getattr(cvt, converter, None)

                if ret is None:
                    # try module converter
                    ret = self.get_converter(converter)

                if ret is None:
                    ret = self.get_resource_clz_by_name(converter)

                if ret is None:
                    ret = self.get_enum_by_name(converter)

                if ret is None:
                    # try parser config
                    ret = self.get(converter)

            if ret is None and converter_str is not None:
                raise ValueError(
                    'Specified converter not supported: {}'.format(
                        converter_str))
        return ret