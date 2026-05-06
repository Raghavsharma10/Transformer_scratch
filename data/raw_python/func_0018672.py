def run_evaluate(self, block: TimeAggregate) -> bool:
        """
        Evaluates the anchor condition against the specified block.
        :param block: Block to run the anchor condition against.
        :return: True, if the anchor condition is met, otherwise, False.
        """
        if self._anchor.evaluate_anchor(block, self._evaluation_context):

            try:
                self.run_reset()
                self._evaluation_context.global_add('anchor', block)
                self._evaluate()
                self._anchor.add_condition_met()
                return True
            finally:
                self._evaluation_context.global_remove('anchor')

        return False