def run_evaluate(self) -> None:
        """
        Overrides the base evaluation to set the value to the evaluation result of the value
        expression in the schema
        """
        result = None
        self.eval_error = False
        if self._needs_evaluation:
            result = self._schema.value.evaluate(self._evaluation_context)

        self.eval_error = result is None
        if self.eval_error:
            return

        # Only set the value if it conforms to the field type
        if not self._schema.is_type_of(result):
            try:
                result = self._schema.type_object(result)
            except Exception as err:
                logging.debug('{} in casting {} to {} for field {}. Error: {}'.format(
                    type(err).__name__, result, self._schema.type,
                    self._schema.fully_qualified_name, err))
                self.eval_error = True
                return

        try:
            result = self._schema.sanitize_object(result)
        except Exception as err:
            logging.debug('{} in sanitizing {} of type {} for field {}. Error: {}'.format(
                type(err).__name__, result, self._schema.type, self._schema.fully_qualified_name,
                err))
            self.eval_error = True
            return

        self.value = result