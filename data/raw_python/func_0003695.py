def set_trace_cond(klass, marker='default', cond=None):
        """ Sets a condition for set_trace statements that have the
            specified marker.  A condition can be either callable, in
            which case it should take one argument, which is the
            number of times set_trace(marker) has been called,
            or it can be a number, in which case the break will
            only be called.
        """
        tc = klass.trace_counts
        tc[marker] = [cond, 0]