def _evaluate_dimension_fields(self) -> bool:
        """
        Evaluates the dimension fields. Returns False if any of the fields could not be evaluated.
        """
        for _, item in self._dimension_fields.items():
            item.run_evaluate()
            if item.eval_error:
                return False
        return True