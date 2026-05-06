def run_evaluate(self, record: Record):
        """
        Evaluates and updates data in the StreamingTransformer.
        :param record: The 'source' record used for the update.
        :raises: IdentityError if identity is different from the one used during
        initialization.
        """
        record_identity = self._schema.get_identity(record)
        if self._identity != record_identity:
            raise IdentityError(
                'Identity in transformer ({}) and new record ({}) do not match'.format(
                    self._identity, record_identity))

        # Add source record and time to the global context
        self._evaluation_context.add_record(record)
        self._evaluation_context.global_add(
            'time',
            DateTimeFieldSchema.sanitize_object(
                self._schema.time.evaluate(self._evaluation_context)))
        super().run_evaluate()

        # Cleanup source and time form the context
        self._evaluation_context.remove_record()
        self._evaluation_context.global_remove('time')