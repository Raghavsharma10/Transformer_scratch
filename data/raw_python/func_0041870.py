def update_arg(self, arg, if_existent=None, **kwargs):
        """
        Update the `add_argument` data for the given parameter

        Parameters
        ----------
        arg: str
            The name of the function argument
        if_existent: bool or None
            If True, the argument is updated. If None (default), the argument
            is only updated, if it exists. Otherwise, if False, the given
            ``**kwargs`` are only used if the argument is not yet existing
        ``**kwargs``
            The keyword arguments any parameter for the
            :meth:`argparse.ArgumentParser.add_argument` method
        """
        if if_existent or (if_existent is None and
                           arg in self.unfinished_arguments):
            self.unfinished_arguments[arg].update(kwargs)
        elif not if_existent and if_existent is not None:
            self.unfinished_arguments.setdefault(arg, kwargs)