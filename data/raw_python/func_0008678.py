def log_transition(self, transition, from_state, instance, *args, **kwargs):
        """Generic transition logging."""
        save = kwargs.pop('save', True)
        log = kwargs.pop('log', True)
        super(Workflow, self).log_transition(
            transition, from_state, instance, *args, **kwargs)
        if save:
            instance.save()
        if log:
            self.db_log(transition, from_state, instance, *args, **kwargs)