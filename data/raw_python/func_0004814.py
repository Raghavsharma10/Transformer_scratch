def main_pred_type(self, value):
        """set main predicate combination type

        :param value: (character) One of ``equals`` (``=``), ``and`` (``&``), ``or`` (``|``),
        ``lessThan`` (``<``), ``lessThanOrEquals`` (``<=``), ``greaterThan`` (``>``),
        ``greaterThanOrEquals`` (``>=``), ``in``, ``within``, ``not`` (``!``), ``like``
        """
        if value not in operators:
            value = operator_lkup.get(value)
        if value:
            self._main_pred_type = value
            self.payload['predicate']['type'] = self._main_pred_type
        else:
            raise Exception("main predicate combiner not a valid operator")