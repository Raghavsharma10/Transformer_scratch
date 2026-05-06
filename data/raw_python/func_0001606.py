def _verify_validates(self, spec, path):
        """Verify thats the 'validates' argument is valid."""
        validates = spec['validates']

        if isinstance(validates, list):
            for validator in validates:
                self._verify_validator(validator, path)
        else:
            self._verify_validator(validates, path)