def json_repr(self, minimal=False):
        """Construct a JSON-friendly representation of the object.

        :param bool minimal: Construct a minimal representation of the object (ignore nulls and empty collections)

        :rtype: dict
        """
        if minimal:
            return {to_camel_case(k): v for k, v in vars(self).items() if (v or v is False or v == 0)}
        else:
            return {to_camel_case(k): v for k, v in vars(self).items()}