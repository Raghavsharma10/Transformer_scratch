def not_(validation_func,  # type: ValidationFuncs
         catch_all=False   # type: bool
         ):
    # type: (...) -> Callable
    """
    Generates the inverse of the provided validation functions: when the validator returns `False` or raises a
    `Failure`, this function returns `True`. Otherwise it raises a `DidNotFail` failure.

    By default, exceptions of types other than `Failure` are not caught and therefore fail the validation
    (`catch_all=False`). To change this behaviour you can turn the `catch_all` parameter to `True`, in which case all
    exceptions will be caught instead of just `Failure`s.

    Note that you may use `not_all(<validation_functions_list>)` as a shortcut for
    `not_(and_(<validation_functions_list>))`

    :param validation_func: the base validation function. A callable, a tuple(callable, help_msg_str),
    a tuple(callable, failure_type), or a list of several such elements. Nested lists
    are supported and indicate an implicit `and_` (such as the main list). Tuples indicate an implicit
    `_failure_raiser`. [mini_lambda](https://smarie.github.io/python-mini-lambda/) expressions can be used instead
    of callables, they will be transformed to functions automatically.
    :param catch_all: an optional boolean flag. By default, only Failure are silently caught and turned into
    a 'ok' result. Turning this flag to True will assume that all exceptions should be caught and turned to a
    'ok' result
    :return:
    """

    def not_v_(x):
        try:
            res = validation_func(x)
            if not result_is_success(res):  # inverse the result
                return True

        except Failure:
            return True  # caught failure: always return True

        except Exception as e:
            if not catch_all:
                raise e
            else:
                return True  # caught exception in 'catch_all' mode: return True

        # if we're here that's a failure
        raise DidNotFail(wrapped_func=validation_func, wrong_value=x, validation_outcome=res)

    not_v_.__name__ = 'not({})'.format(get_callable_name(validation_func))
    return not_v_