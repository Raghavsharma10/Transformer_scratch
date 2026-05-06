def assert_valid(self,
                     name,             # type: str
                     value,            # type: Any
                     error_type=None,  # type: Type[ValidationError]
                     help_msg=None,    # type: str
                     **kw_context_args):
        """
        Asserts that the provided named value is valid with respect to the inner base validation functions. It returns
        silently in case of success, and raises a `ValidationError` or a subclass in case of failure. This corresponds
        to a 'Defensive programming' (sometimes known as 'Offensive programming') mode.

        By default this raises instances of `ValidationError` with a default message, in case of failure. There are
        two ways that you can customize this behaviour:

         * if you set `help_msg` in this method or in `Validator` constructor, instances of `ValidationError` created
         will be customized with the provided help message.

         * if you set `error_type` in this method or in `Validator` constructor, instances of your custom class will be
         created. Note that you may still provide a `help_msg`.

        It is recommended that Users define their own validation error types (case 2 above), so as to provide a unique
        error type for each kind of applicative error. This eases the process of error handling at app-level.

        :param name: the name of the variable to validate (for error messages)
        :param value: the value to validate
        :param error_type: a subclass of `ValidationError` to raise in case of validation failure. By default a
            `ValidationError` will be raised with the provided `help_msg`
        :param help_msg: an optional help message to be used in the raised error in case of validation failure.
        :param kw_context_args: optional contextual information to store in the exception, and that may be also used
            to format the help message
        :return: nothing in case of success. Otherwise, raises a ValidationError
        """
        try:
            # perform validation
            res = self.main_function(value)

        except Exception as e:
            # caught any exception: raise ValidationError or subclass with that exception in the details
            # --old bad idea: first wrap into a failure ==> NO !!! I tried and it was making it far too messy/verbose

            # note: we do not have to 'raise x from e' of `raise_from`since the ValidationError constructor already
            # sets the __cause__ so we can safely take the same handling than for non-exception failures.
            res = e

        # check the result
        if not result_is_success(res):
            raise_(self._create_validation_error(name, value, validation_outcome=res, error_type=error_type,
                                                 help_msg=help_msg, **kw_context_args))