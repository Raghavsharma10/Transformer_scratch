def _freeze(self) -> OrderedDict:
        """
        Evaluate all of the column values and return the result
        :return: column/value tuples
        """
        return OrderedDict(**{k: getattr(self, k, None) for k in super().__getattribute__("_columns")})