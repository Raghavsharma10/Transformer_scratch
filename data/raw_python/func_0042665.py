def validate(self, config):
        """Validate the given config against the `Scheme`.

        Args:
            config (dict): The configuration to validate.

        Raises:
            errors.SchemeValidationError: The configuration fails
                validation against the `Schema`.
        """
        if not isinstance(config, dict):
            raise errors.SchemeValidationError(
                'Scheme can only validate a dictionary config, but was given '
                '{} (type: {})'.format(config, type(config))
            )

        for arg in self.args:
            # the option exists in the config
            if arg.name in config:
                arg.validate(config[arg.name])

            # the option does not exist in the config
            else:
                # if the option is not required, then it is fine to omit.
                # otherwise, its omission constitutes a validation error.
                if arg.required:
                    raise errors.SchemeValidationError(
                        'Option "{}" is required, but not found.'.format(arg.name)
                    )