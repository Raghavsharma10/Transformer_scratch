def validate(self, syntax_only=False):
        """Validates a Packer Template file (`packer validate`)

        If the validation failed, an `sh` exception will be raised.
        :param bool syntax_only: Whether to validate the syntax only
        without validating the configuration itself.
        """
        self.packer_cmd = self.packer.validate

        self._add_opt('-syntax-only' if syntax_only else None)
        self._append_base_arguments()
        self._add_opt(self.packerfile)

        # as sh raises an exception rather than return a value when execution
        # fails we create an object to return the exception and the validation
        # state
        try:
            validation = self.packer_cmd()
            validation.succeeded = validation.exit_code == 0
            validation.error = None
        except Exception as ex:
            validation = ValidationObject()
            validation.succeeded = False
            validation.failed = True
            validation.error = ex.message
        return validation