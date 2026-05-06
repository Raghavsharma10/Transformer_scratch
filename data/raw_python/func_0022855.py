def arg_to_array(func):
    """
    Decorator to convert argument to array.

    Parameters
    ----------
    func : function
        The function to decorate.

    Returns
    -------
    func : function
        The decorated function.
    """
    def fn(self, arg, *args, **kwargs):
        """Function

        Parameters
        ----------
        arg : array-like
            Argument to convert.
        *args : tuple
            Arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        value : object
            The return value of the function.
        """
        return func(self, np.array(arg), *args, **kwargs)
    return fn