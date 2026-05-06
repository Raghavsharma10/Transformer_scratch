def _check_local_handlers(cls, signals, handlers, namespace, configs):
        """For every marked handler, see if there is a suitable signal. If
        not, raise an error."""
        for aname, sig_name in handlers.items():
            # WARN: this code doesn't take in account the case where a new
            # method with the same name of an handler in a base class is
            # present in this class but it isn't an handler (so the handler
            # with the same name should be removed from the handlers)
            if sig_name not in signals:
                disable_check = configs[aname].get('disable_check', False)
                if not disable_check:
                    raise SignalError("Cannot find a signal named '%s'"
                                      % sig_name)