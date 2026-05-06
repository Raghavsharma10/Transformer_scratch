def activate(self):
        """
        Whenever L{Scheduler} or L{SubScheduler} is created, either newly or
        when loaded from a database, emit a deprecation warning referring
        people to L{IScheduler}.
        """
        # This is unfortunate.  Perhaps it is the best thing which works (it is
        # the first I found). -exarkun
        if '_axiom_memory_dummy' in vars(self):
            stacklevel = 7
        else:
            stacklevel = 5
        warnings.warn(
            self.__class__.__name__ + " is deprecated since Axiom 0.5.32.  "
            "Just adapt stores to IScheduler.",
            category=PendingDeprecationWarning,
            stacklevel=stacklevel)