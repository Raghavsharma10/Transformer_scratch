def settrace(self, do_set):
        """Set or remove the trace function."""
        if do_set:
            sys.settrace(self.trace_dispatch)
        else:
            sys.settrace(None)