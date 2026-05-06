def user_func(func, arg_types=None, return_type=None):
    """Create an EFILTER-callable version of function 'func'.

    As a security precaution, EFILTER will not execute Python callables
    unless they implement the IApplicative protocol. There is a perfectly good
    implementation of this protocol in the standard library and user functions
    can inherit from it.

    This will declare a subclass of the standard library TypedFunction and
    return an instance of it that EFILTER will happily call.

    Arguments:
        func: A Python callable that will serve as the implementation.
        arg_types (optional): A tuple of argument types. If the function takes
            keyword arguments, they must still have a defined order.
        return_type (optional): The type the function returns.

    Returns:
        An instance of a custom subclass of efilter.stdlib.core.TypedFunction.

    Examples:
        def my_callback(tag):
            print("I got %r" % tag)

        api.apply("if True then my_callback('Hello World!')",
                  vars={
                    "my_callback": api.user_func(my_callback)
                  })

        # This should print "I got 'Hello World!'".
    """
    class UserFunction(std_core.TypedFunction):
        name = func.__name__

        def __call__(self, *args, **kwargs):
            return func(*args, **kwargs)

        @classmethod
        def reflect_static_args(cls):
            return arg_types

        @classmethod
        def reflect_static_return(cls):
            return return_type

    return UserFunction()