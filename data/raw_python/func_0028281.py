def build_ctx(pythonpath=None):
    """
    Decorator that makes decorated function use BuildContext instead of \
    Context instance. BuildContext instance has more methods.

    :param pythonpath: Path or list of paths to add to environment variable
        PYTHONPATH. Each path can be absolute path, or relative path relative
        to top directory.

        Notice if this decorator is used without arguments, argument
        `pythonpath` is the decorated function.

    :return: Two situations:

        - If decorator arguments are given, return no-argument decorator.
        - If decorator arguments are not given, return wrapper function.
    """
    # If argument `pythonpath` is string
    if isinstance(pythonpath, str):
        # Create paths list containing the string
        path_s = [pythonpath]

    # If argument `pythonpath` is list
    elif isinstance(pythonpath, list):
        # Use the list as paths list
        path_s = pythonpath

    # If argument `pythonpath` is not string or list,
    # it means the decorator is used without arguments.
    else:
        # Set paths list be None
        path_s = None

    # Create no-argument decorator
    def _noarg_decorator(func):
        """
        No-argument decorator.

        :param func: Decorated function.

        :return: Wrapper function.
        """
        # Create BuildContext subclass
        class _BuildContext(BuildContext):
            # Set command name for the context class
            cmd = func.__name__

            # Set function name for the context class
            fun = func.__name__

        # Create wrapper function
        @wraps(func)
        def _new_func(ctx, *args, **kwargs):
            """
            Wrapper function.

            :param ctx: BuildContext object.

            :param \\*args: Other arguments passed to decorated function.

            :param \\*\\*kwargs: Other keyword arguments passed to decorated
            function.

            :return: Decorated function's call result.
            """
            # If paths list is not empty
            if path_s:
                # For each path
                for path in path_s:
                    # If the path is absolute path
                    if os.path.isabs(path):
                        # Use the path as absolute path
                        abs_path = path

                    # If the path is not absolute path,
                    # it means relative path relative to top directory.
                    else:
                        # Create path node
                        path_node = create_node(ctx, path)

                        # Get absolute path
                        abs_path = path_node.abspath()

                    # Add the absolute path to environment variable PYTHONPATH
                    add_pythonpath(abs_path)

            # Call the decorated function
            result = func(ctx, *args, **kwargs)

            # Return the call result
            return result

        # Store the created context class with the wrapper function
        _new_func._context_class = _BuildContext  # pylint: disable=W0212

        # Return the wrapper function
        return _new_func

    # If decorator arguments are given
    if path_s is not None:
        # Return no-argument decorator
        return _noarg_decorator

    # If decorator arguments are not given
    else:
        # Argument `pythonpath` is the decorated function
        _func = pythonpath

        # Call the no-argument decorator to create wrapper function
        wrapper_func = _noarg_decorator(_func)

        # Return the wrapper function
        return wrapper_func