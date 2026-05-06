def build_expression(self, attribute: str) -> Optional[Expression]:
        """ Builds an expression object.  Adds an error if expression creation has errors. """

        expression_string = self._spec.get(attribute, None)
        if expression_string:
            try:
                return Expression(str(expression_string))
            except Exception as err:
                self.add_errors(
                    InvalidExpressionError(self.fully_qualified_name, self._spec, attribute, err))

        return None