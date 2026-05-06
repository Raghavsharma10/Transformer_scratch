def instance_ik_model_receiver(fn):
    """
    A method decorator that filters out sign_original_specals coming from models that don't
    have fields that function as ImageFieldSourceGroup sources.

    """
    @wraps(fn)
    def receiver(self, sender, **kwargs):
        # print 'inspect.isclass(sender? %s'%(inspect.isclass(sender))
        if not inspect.isclass(sender):
            return
        for src in self._source_groups:
            if issubclass(sender, src.model_class):
                fn(self, sender=sender, **kwargs)

                # If we find a match, return. We don't want to handle the signal
                # more than once.
                return
    return receiver