def excmessage_decorator(description) -> Callable:
    """Wrap a function with |augment_excmessage|.

    Function |excmessage_decorator| is a means to apply function
    |augment_excmessage| more efficiently.  Suppose you would apply
    function |augment_excmessage| in a function that adds and returns
    to numbers:

    >>> from  hydpy.core import objecttools
    >>> def add(x, y):
    ...     try:
    ...         return x + y
    ...     except BaseException:
    ...         objecttools.augment_excmessage(
    ...             'While trying to add `x` and `y`')

    This works as excepted...

    >>> add(1, 2)
    3
    >>> add(1, [])
    Traceback (most recent call last):
    ...
    TypeError: While trying to add `x` and `y`, the following error \
occurred: unsupported operand type(s) for +: 'int' and 'list'

    ...but can be achieved with much less code using |excmessage_decorator|:

    >>> @objecttools.excmessage_decorator(
    ...     'add `x` and `y`')
    ... def add(x, y):
    ...     return x+y

    >>> add(1, 2)
    3

    >>> add(1, [])
    Traceback (most recent call last):
    ...
    TypeError: While trying to add `x` and `y`, the following error \
occurred: unsupported operand type(s) for +: 'int' and 'list'

    Additionally, exception messages related to wrong function calls
    are now also augmented:

    >>> add(1)
    Traceback (most recent call last):
    ...
    TypeError: While trying to add `x` and `y`, the following error \
occurred: add() missing 1 required positional argument: 'y'

    |excmessage_decorator| evaluates the given string like an f-string,
    allowing to mention the argument values of the called function and
    to make use of all string modification functions provided by modules
    |objecttools|:

    >>> @objecttools.excmessage_decorator(
    ...     'add `x` ({repr_(x, 2)}) and `y` ({repr_(y, 2)})')
    ... def add(x, y):
    ...     return x+y

    >>> add(1.1111, 'wrong')
    Traceback (most recent call last):
    ...
    TypeError: While trying to add `x` (1.11) and `y` (wrong), the following \
error occurred: unsupported operand type(s) for +: 'float' and 'str'
    >>> add(1)
    Traceback (most recent call last):
    ...
    TypeError: While trying to add `x` (1) and `y` (?), the following error \
occurred: add() missing 1 required positional argument: 'y'
    >>> add(y=1)
    Traceback (most recent call last):
    ...
    TypeError: While trying to add `x` (?) and `y` (1), the following error \
occurred: add() missing 1 required positional argument: 'x'

    Apply |excmessage_decorator| on methods also works fine:

    >>> class Adder:
    ...     def __init__(self):
    ...         self.value = 0
    ...     @objecttools.excmessage_decorator(
    ...         'add an instance of class `{classname(self)}` with value '
    ...         '`{repr_(other, 2)}` of type `{classname(other)}`')
    ...     def __iadd__(self, other):
    ...         self.value += other
    ...         return self

    >>> adder = Adder()
    >>> adder += 1
    >>> adder.value
    1
    >>> adder += 'wrong'
    Traceback (most recent call last):
    ...
    TypeError: While trying to add an instance of class `Adder` with value \
`wrong` of type `str`, the following error occurred: unsupported operand \
type(s) for +=: 'int' and 'str'

    It is made sure that no information of the decorated function is lost:

    >>> add.__name__
    'add'
    """
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        """Apply |augment_excmessage| when the wrapped function fails."""
        # pylint: disable=unused-argument
        try:
            return wrapped(*args, **kwargs)
        except BaseException:
            info = kwargs.copy()
            info['self'] = instance
            argnames = inspect.getfullargspec(wrapped).args
            if argnames[0] == 'self':
                argnames = argnames[1:]
            for argname, arg in zip(argnames, args):
                info[argname] = arg
            for argname in argnames:
                if argname not in info:
                    info[argname] = '?'
            message = eval(
                f"f'While trying to {description}'", globals(), info)
            augment_excmessage(message)
    return wrapper