def _create_validation_error(self,
                                 name,                     # type: str
                                 value,                    # type: Any
                                 validation_outcome=None,  # type: Any
                                 error_type=None,          # type: Type[ValidationError]
                                 help_msg=None,            # type: str
                                 **kw_context_args):
        """ The function doing the final error raising.  """

        # first merge the info provided in arguments and in self
        error_type = error_type or self.error_type
        help_msg = help_msg or self.help_msg
        ctx = copy(self.kw_context_args)
        ctx.update(kw_context_args)

        # allow the class to override the name
        name = self._get_name_for_errors(name)

        if issubclass(error_type, TypeError) or issubclass(error_type, ValueError):
            # this is most probably a custom error type, it is already annotated with ValueError and/or TypeError
            # so use it 'as is'
            new_error_type = error_type
        else:
            # Add the appropriate TypeError/ValueError base type dynamically
            additional_type = None
            if isinstance(validation_outcome, Exception):
                if is_error_of_type(validation_outcome, TypeError):
                    additional_type = TypeError
                elif is_error_of_type(validation_outcome, ValueError):
                    additional_type = ValueError
            if additional_type is None:
                # not much we can do here, let's assume a ValueError, that is more probable
                additional_type = ValueError

            new_error_type = add_base_type_dynamically(error_type, additional_type)

        # then raise the appropriate ValidationError or subclass
        return new_error_type(validator=self, var_value=value, var_name=name, validation_outcome=validation_outcome,
                              help_msg=help_msg, **ctx)