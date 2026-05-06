def drop_prefix_and_return_type(function):
    """Takes the function value from a frame and drops prefix and return type

    For example::

        static void * Allocator<MozJemallocBase>::malloc(unsigned __int64)
        ^      ^^^^^^ return type
        prefix

    This gets changes to this::

        Allocator<MozJemallocBase>::malloc(unsigned __int64)

    This tokenizes on space, but takes into account types, generics, traits,
    function arguments, and other parts of the function signature delimited by
    things like `', <>, {}, [], and () for both C/C++ and Rust.

    After tokenizing, this returns the last token since that's comprised of the
    function name and its arguments.

    :arg function: the function value in a frame to drop bits from

    :returns: adjusted function value

    """
    DELIMITERS = {
        '(': ')',
        '{': '}',
        '[': ']',
        '<': '>',
        '`': "'"
    }
    OPEN = DELIMITERS.keys()
    CLOSE = DELIMITERS.values()

    # The list of tokens accumulated so far
    tokens = []

    # Keeps track of open delimiters so we can match and close them
    levels = []

    # The current token we're building
    current = []

    for i, char in enumerate(function):
        if char in OPEN:
            levels.append(char)
            current.append(char)
        elif char in CLOSE:
            if levels and DELIMITERS[levels[-1]] == char:
                levels.pop()
                current.append(char)
            else:
                # This is an unmatched close.
                current.append(char)
        elif levels:
            current.append(char)
        elif char == ' ':
            tokens.append(''.join(current))
            current = []
        else:
            current.append(char)

    if current:
        tokens.append(''.join(current))

    while len(tokens) > 1 and tokens[-1].startswith(('(', '[clone')):
        # It's possible for the function signature to have a space between
        # the function name and the parenthesized arguments or [clone ...]
        # thing. If that's the case, we join the last two tokens. We keep doing
        # that until the last token is nice.
        #
        # Example:
        #
        #     somefunc (int arg1, int arg2)
        #             ^
        #     somefunc(int arg1, int arg2) [clone .cold.111]
        #                                 ^
        #     somefunc(int arg1, int arg2) [clone .cold.111] [clone .cold.222]
        #                                 ^                 ^
        tokens = tokens[:-2] + [' '.join(tokens[-2:])]

    return tokens[-1]