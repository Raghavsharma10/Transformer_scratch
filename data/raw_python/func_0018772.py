def merge(self, evaluation_context: 'EvaluationContext') -> None:
        """
        Merges the provided evaluation context to the current evaluation context.
        :param evaluation_context: Evaluation context to merge.
        """
        self.global_context.merge(evaluation_context.global_context)
        self.local_context.merge(evaluation_context.local_context)