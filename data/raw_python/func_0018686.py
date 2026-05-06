def _needs_evaluation(self) -> bool:
        """
        Returns True when:
            1. Where clause is not specified
            2. Where WHERE clause is specified and it evaluates to True
        Returns false if a where clause is specified and it evaluates to False
        """
        return self._schema.when is None or self._schema.when.evaluate(self._evaluation_context)