def decorate_cls_with_validation(cls,
                                 field_name,        # type: str
                                 *validation_func,  # type: ValidationFuncs
                                 **kwargs):
    # type: (...) -> Type[Any]
    """
    This method is equivalent to decorating a class with the `@validate_field` decorator but can be used a posteriori.

    :param cls: the class to decorate
    :param field_name: the name of the argument to validate or _OUT_KEY for output validation
    :param validation_func: the validation function or
        list of validation functions to use. A validation function may be a callable, a tuple(callable, help_msg_str),
        a tuple(callable, failure_type), or a list of several such elements. Nested lists are supported and indicate an
        implicit `and_` (such as the main list). Tuples indicate an implicit `_failure_raiser`.
        [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead of callables, they
        will be transformed to functions automatically.
    :param error_type: a subclass of ValidationError to raise in case of validation failure. By default a
        ValidationError will be raised with the provided help_msg
    :param help_msg: an optional help message to be used in the raised error in case of validation failure.
    :param none_policy: describes how None values should be handled. See `NoneArgPolicy` for the various possibilities.
        Default is `NoneArgPolicy.ACCEPT_IF_OPTIONAl_ELSE_REJECT`.
    :param kw_context_args: optional contextual information to store in the exception, and that may be also used
        to format the help message
    :return: the decorated function, that will perform input validation (using `_assert_input_is_valid`) before
        executing the function's code everytime it is executed.
    """
    error_type, help_msg, none_policy = pop_kwargs(kwargs, [('error_type', None),
                                                            ('help_msg', None),
                                                            ('none_policy', None)], allow_others=True)
    # the rest of keyword arguments is used as context.
    kw_context_args = kwargs

    if not isclass(cls):
        raise TypeError('decorated cls should be a class')

    if hasattr(cls, field_name):
        # ** A class field with that name exist. Is it a descriptor ?

        var = cls.__dict__[field_name]  # note: we cannot use getattr here

        if hasattr(var, '__set__') and callable(var.__set__):

            if isinstance(var, property):
                # *** OLD WAY which was losing type hints and default values (see var.__set__ signature) ***
                # properties are special beasts: their methods are method-wrappers (CPython) and can not have properties
                # so we have to create a wrapper (sic) before sending it to the main wrapping function
                # def func(inst, value):
                #     var.__set__(inst, value)

                # *** NEW WAY : more elegant, use directly the setter provided by the user ***
                func = var.fset
                nb_args = 2
            elif ismethod(var.__set__):
                # bound method: normal. Let's access to the underlying function
                func = var.__set__.__func__
                nb_args = 3
            else:
                # strange.. but lets try to continue
                func = var.__set__
                nb_args = 3

            # retrieve target function signature, check it and retrieve the 3d param
            # since signature is "def __set__(self, obj, val)"
            func_sig = signature(func)
            if len(func_sig.parameters) != nb_args:
                raise ValueError("Class field '{}' is a valid class descriptor for class '{}' but it does not implement"
                                 " __set__ with the correct number of parameters, so it is not possible to add "
                                 "validation to it. See https://docs.python.org/3.6/howto/descriptor.html".
                                 format(field_name, cls.__name__))
            # extract the correct name
            descriptor_arg_name = list(func_sig.parameters.items())[-1][0]

            # do the same than in decorate_with_validation but with a class field validator
            # new_setter = decorate_with_validation(func, descriptor_arg_name, *validation_func, help_msg=help_msg,
            #                                       error_type=error_type, none_policy=none_policy,
            #                                       _clazz_field_name_=field_name, **kw_context_args)

            # --create the new validator
            none_policy = none_policy or NoneArgPolicy.SKIP_IF_NONABLE_ELSE_VALIDATE
            new_validator = _create_function_validator(func, func_sig, descriptor_arg_name, *validation_func,
                                                       none_policy=none_policy, error_type=error_type,
                                                       help_msg=help_msg,
                                                       validated_class=cls, validated_class_field_name=field_name,
                                                       **kw_context_args)

            # -- create the new setter with validation
            new_setter = decorate_with_validators(func, func_signature=func_sig, **{descriptor_arg_name: new_validator})

            # replace the old one
            if isinstance(var, property):
                # properties are special beasts 2
                setattr(cls, field_name, var.setter(new_setter))
            else:
                # do not use type() for python 2 compat
                var.__class__.__set__ = new_setter

        elif (hasattr(var, '__get__') and callable(var.__get__)) \
            or (hasattr(var, '__delete__') and callable(var.__delete__)):
            # this is a descriptor but it does not have any setter method: impossible to validate
            raise ValueError("Class field '{}' is a valid class descriptor for class '{}' but it does not implement "
                             "__set__ so it is not possible to add validation to it. See "
                             "https://docs.python.org/3.6/howto/descriptor.html".format(field_name, cls.__name__))

        else:
            # this is not a descriptor: unsupported
            raise ValueError("Class field '{}.{}' is not a valid class descriptor, see "
                             "https://docs.python.org/3.6/howto/descriptor.html".format(cls.__name__, field_name))

    else:
        # ** No class field with that name exist

        # ? check for attrs ? > no specific need anymore, this is the same than annotating the constructor
        # if hasattr(cls, '__attrs_attrs__'): this was a proof of attrs-defined class

        # try to annotate the generated constructor
        try:
            init_func = cls.__init__
            if sys.version_info < (3, 0):
                try:
                    # python 2 - we have to access the inner `im_func`
                    init_func = cls.__init__.im_func
                except AttributeError:
                    pass

            cls.__init__ = decorate_with_validation(init_func, field_name, *validation_func, help_msg=help_msg,
                                                    _constructor_of_cls_=cls,
                                                    error_type=error_type, none_policy=none_policy, **kw_context_args)

        except InvalidNameError:
            # the field was not found

            # TODO should we also check if a __setattr__ is defined ?
            # (for __setattr__ see https://stackoverflow.com/questions/15750522/class-properties-and-setattr/15751159)

            # finally raise an error
            raise ValueError("@validate_field definition exception: field '{}' can not be found in class '{}', and it "
                             "is also not an input argument of the __init__ method.".format(field_name, cls.__name__))

    return cls