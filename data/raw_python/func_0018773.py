def evaluate(self, evaluation_context: EvaluationContext) -> Any:
        """
        Evaluates the expression with the context provided.  If the execution
        results in failure, an ExpressionEvaluationException encapsulating the
        underlying exception is raised.
        :param evaluation_context: Global and local context dictionary to be passed for evaluation
        """
        try:

            if self.type == ExpressionType.EVAL:
                return eval(self.code_object, evaluation_context.global_context,
                            evaluation_context.local_context)

            elif self.type == ExpressionType.EXEC:
                return exec(self.code_object, evaluation_context.global_context,
                            evaluation_context.local_context)

        except Exception as err:
            # Evaluation exceptions are expected because of missing fields in the source 'Record'.
            logging.debug('{} in evaluating expression {}. Error: {}'.format(
                type(err).__name__, self.code_string, err))
            # These should result in an exception being raised:
            # NameError - Exceptions thrown because of using names in the expression which are not
            #   present in EvaluationContext. A common cause for this is typos in the BTS.
            # MissingAttributeError - Exception thrown when a BTS nested item is used which does not
            #   exist. Should only happen for erroneous BTSs.
            # ImportError - Thrown when there is a failure in importing other modules.
            if isinstance(err, (NameError, MissingAttributeError, ImportError)):
                raise err
            return None