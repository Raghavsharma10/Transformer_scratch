def _get_calling_module(self):
        """Get the last module in the call stack that is not this module or ``None`` if
        the call originated from this module.

        """
        for frame in inspect.stack():
            mod = inspect.getmodule(frame[0])
            logger.debug(f'calling module: {mod}')
            if mod is not None:
                mod_name = mod.__name__
                if mod_name != __name__:
                    return mod