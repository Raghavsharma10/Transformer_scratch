def emit_only(self, event: str, func_names: Union[str, List[str]], *args,
                  **kwargs) -> None:
        """ Specifically only emits certain subscribed events.

        :param event: Name of the event.
        :type event: str

        :param func_names: Function(s) to emit.
        :type func_names: Union[ str | List[str] ]
        """
        if isinstance(func_names, str):
            func_names = [func_names]

        for func in self._event_funcs(event):
            if func.__name__ in func_names:
                func(*args, **kwargs)