def validate(name,                   # type: str
             value,                  # type: Any
             enforce_not_none=True,  # type: bool
             equals=None,            # type: Any
             instance_of=None,       # type: Union[Type, Tuple[Type]]
             subclass_of=None,       # type: Union[Type, Tuple[Type]]
             is_in=None,             # type: Container
             subset_of=None,         # type: Set
             contains = None,        # type: Union[Any, Iterable]
             superset_of=None,       # type: Set
             min_value=None,         # type: Any
             min_strict=False,       # type: bool
             max_value=None,         # type: Any
             max_strict=False,       # type: bool
             length=None,            # type: int
             min_len=None,           # type: int
             min_len_strict=False,   # type: bool
             max_len=None,           # type: int
             max_len_strict=False,   # type: bool
             custom=None,            # type: Callable[[Any], Any]
             error_type=None,        # type: Type[ValidationError]
             help_msg=None,          # type: str
             **kw_context_args):
    """
    A validation function for quick inline validation of `value`, with minimal capabilities:

    * None handling: reject None (enforce_not_none=True, default), or accept None silently (enforce_not_none=False)
    * Type validation: `value` should be an instance of any of `var_types` if provided
    * Value validation:
       * if `allowed_values` is provided, `value` should be in that set
       * if `min_value` (resp. `max_value`) is provided, `value` should be greater than it. Comparison is not strict by
       default and can be set to strict by setting `min_strict`, resp. `max_strict`, to `True`
       * if `min_len` (resp. `max_len`) is provided, `len(value)` should be greater than it. Comparison is not strict by
       default and can be set to strict by setting `min_len_strict`, resp. `max_len_strict`, to `True`

    :param name: the applicative name of the checked value, that will be used in error messages
    :param value: the value to check
    :param enforce_not_none: boolean, default True. Whether to enforce that `value` is not None.
    :param equals: an optional value to enforce.
    :param instance_of: optional type(s) to enforce. If a tuple of types is provided it is considered alternate types: one
        match is enough to succeed. If None, type will not be enforced
    :param subclass_of: optional type(s) to enforce. If a tuple of types is provided it is considered alternate types: one
        match is enough to succeed. If None, type will not be enforced
    :param is_in: an optional set of allowed values.
    :param subset_of: an optional superset for the variable
    :param contains: an optional value that the variable should contain (value in variable == True)
    :param superset_of: an optional subset for the variable
    :param min_value: an optional minimum value
    :param min_strict: if True, only values strictly greater than `min_value` will be accepted
    :param max_value: an optional maximum value
    :param max_strict: if True, only values strictly lesser than `max_value` will be accepted
    :param length: an optional strict length
    :param min_len: an optional minimum length
    :param min_len_strict: if True, only values with length strictly greater than `min_len` will be accepted
    :param max_len: an optional maximum length
    :param max_len_strict: if True, only values with length strictly lesser than `max_len` will be accepted
    :param custom: a custom base validation function or list of base validation functions to use. This is the same
        syntax than for valid8 decorators. A callable, a tuple(callable, help_msg_str), a tuple(callable, failure_type),
        or a list of several such elements. Nested lists are supported and indicate an implicit `and_`. Tuples indicate
        an implicit `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be
        used instead of callables, they will be transformed to functions automatically.
    :param error_type: a subclass of `ValidationError` to raise in case of validation failure. By default a
        `ValidationError` will be raised with the provided `help_msg`
    :param help_msg: an optional help message to be used in the raised error in case of validation failure.
    :param kw_context_args: optional contextual information to store in the exception, and that may be also used
        to format the help message
    :return: nothing in case of success. Otherwise, raises a ValidationError
    """

    # backwards compatibility
    instance_of = instance_of or (kw_context_args.pop('allowed_types') if 'allowed_types' in kw_context_args else None)
    is_in = is_in or (kw_context_args.pop('allowed_values') if 'allowed_values' in kw_context_args else None)

    try:
        # the following corresponds to an inline version of
        # - _none_rejecter in base.py
        # - gt/lt in comparables.py
        # - is_in/contains/subset_of/superset_of/has_length/minlen/maxlen/is_in in collections.py
        # - instance_of/subclass_of in types.py

        # try (https://github.com/orf/inliner) to perform the inlining below automatically without code duplication ?
        # > maybe not because quite dangerous (AST mod) and below we skip the "return True" everywhere for performance
        #
        # Another alternative: easy Cython compiling https://github.com/AlanCristhian/statically
        # > but this is not py2 compliant

        if value is None:
            # inlined version of _none_rejecter in base.py
            if enforce_not_none:
                raise ValueIsNone(wrong_value=value)
                # raise MissingMandatoryParameterException('Error, ' + name + '" is mandatory, it should be non-None')

            # else do nothing and return

        else:
            if equals is not None:
                if value != equals:
                    raise NotEqual(wrong_value=value, ref_value=equals)

            if instance_of is not None:
                assert_instance_of(value, instance_of)

            if subclass_of is not None:
                assert_subclass_of(value, subclass_of)

            if is_in is not None:
                # inlined version of is_in(allowed_values=allowed_values)(value) without 'return True'
                if value not in is_in:
                    raise NotInAllowedValues(wrong_value=value, allowed_values=is_in)

            if contains is not None:
                # inlined version of contains(ref_value=contains)(value) without 'return True'
                if contains not in value:
                    raise DoesNotContainValue(wrong_value=value, ref_value=contains)

            if subset_of is not None:
                # inlined version of is_subset(reference_set=subset_of)(value)
                missing = value - subset_of
                if len(missing) != 0:
                    raise NotSubset(wrong_value=value, reference_set=subset_of, unsupported=missing)

            if superset_of is not None:
                # inlined version of is_superset(reference_set=superset_of)(value)
                missing = superset_of - value
                if len(missing) != 0:
                    raise NotSuperset(wrong_value=value, reference_set=superset_of, missing=missing)

            if min_value is not None:
                # inlined version of gt(min_value=min_value, strict=min_strict)(value) without 'return True'
                if min_strict:
                    if not value > min_value:
                        raise TooSmall(wrong_value=value, min_value=min_value, strict=True)
                else:
                    if not value >= min_value:
                        raise TooSmall(wrong_value=value, min_value=min_value, strict=False)

            if max_value is not None:
                # inlined version of lt(max_value=max_value, strict=max_strict)(value) without 'return True'
                if max_strict:
                    if not value < max_value:
                        raise TooBig(wrong_value=value, max_value=max_value, strict=True)
                else:
                    if not value <= max_value:
                        raise TooBig(wrong_value=value, max_value=max_value, strict=False)

            if length is not None:
                # inlined version of has_length() without 'return True'
                if len(value) != length:
                    raise WrongLength(wrong_value=value, ref_length=length)

            if min_len is not None:
                # inlined version of minlen(min_length=min_len, strict=min_len_strict)(value) without 'return True'
                if min_len_strict:
                    if not len(value) > min_len:
                        raise TooShort(wrong_value=value, min_length=min_len, strict=True)
                else:
                    if not len(value) >= min_len:
                        raise TooShort(wrong_value=value, min_length=min_len, strict=False)

            if max_len is not None:
                # inlined version of maxlen(max_length=max_len, strict=max_len_strict)(value) without 'return True'
                if max_len_strict:
                    if not len(value) < max_len:
                        raise TooLong(wrong_value=value, max_length=max_len, strict=True)
                else:
                    if not len(value) <= max_len:
                        raise TooLong(wrong_value=value, max_length=max_len, strict=False)

    except Exception as e:
        err = _QUICK_VALIDATOR._create_validation_error(name, value, validation_outcome=e, error_type=error_type,
                                                        help_msg=help_msg, **kw_context_args)
        raise_(err)

    if custom is not None:
        # traditional custom validator
        assert_valid(name, value, custom, error_type=error_type, help_msg=help_msg, **kw_context_args)
    else:
        # basic (and not enough) check to verify that there was no typo leading an argument to be put in kw_context_args
        if error_type is None and help_msg is None and len(kw_context_args) > 0:
            raise ValueError("Keyword context arguments have been provided but help_msg and error_type are not: {}"
                             "".format(kw_context_args))