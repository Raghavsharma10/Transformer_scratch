def create_manually(cls,
                        validation_function_name,  # type: str
                        var_name,                  # type: str
                        var_value,
                        validation_outcome=None,   # type: Any
                        help_msg=None,             # type: str
                        append_details=True,       # type: bool
                        **kw_context_args):
        """
        Creates an instance without using a Validator.

        This method is not the primary way that errors are created - they should rather created by the validation entry
        points. However it can be handy in rare edge cases.

        :param validation_function_name:
        :param var_name:
        :param var_value:
        :param validation_outcome:
        :param help_msg:
        :param append_details:
        :param kw_context_args:
        :return:
        """
        # create a dummy validator
        def val_fun(x):
            pass
        val_fun.__name__ = validation_function_name
        validator = Validator(val_fun, error_type=cls, help_msg=help_msg, **kw_context_args)

        # create the exception
        # e = cls(validator, var_value, var_name, validation_outcome=validation_outcome, help_msg=help_msg,
        #         append_details=append_details, **kw_context_args)
        e = validator._create_validation_error(var_name, var_value, validation_outcome, error_type=cls,
                                               help_msg=help_msg, **kw_context_args)
        return e