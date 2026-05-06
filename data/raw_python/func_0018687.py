def run_evaluate(self, *args, **kwargs) -> None:
        """
        Evaluates the current item
        :returns An evaluation result object containing the result, or reasons why
        evaluation failed
        """
        if self._needs_evaluation:
            for _, item in self._nested_items.items():
                item.run_evaluate()