def check_dict_expected_keys(self, expected_keys, current, dict_name):
        """ Check that we don't have unknown keys in a dictionary.

        It does not raise an error if we have less keys than expected.
        """
        if not isinstance(current, dict):
            raise ParseError(u"'{}' key must be a dict".format(dict_name),
                             YAML_EXAMPLE)
        expected_keys = set(expected_keys)
        current_keys = {key for key in current}
        extra_keys = current_keys - expected_keys
        if extra_keys:
            message = u"{}: the keys {} are unexpected. (allowed keys: {})"
            raise ParseError(
                message.format(
                    dict_name,
                    list(extra_keys),
                    list(expected_keys),
                ),
                YAML_EXAMPLE,
            )