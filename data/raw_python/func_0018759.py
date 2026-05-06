def get_identity(self, record: Record) -> str:
        """
        Evaluates and returns the identity as specified in the schema.
        :param record: Record which is used to determine the identity.
        :return: The evaluated identity
        :raises: IdentityError if identity cannot be determined.
        """
        context = self.schema_context.context
        context.add_record(record)
        identity = self.identity.evaluate(context)
        if not identity:
            raise IdentityError('Could not determine identity using {}. Record is {}'.format(
                self.identity.code_string, record))
        context.remove_record()
        return identity