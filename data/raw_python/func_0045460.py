def process_objects(kls):
        """
        Applies default Meta properties.
        """
        # first add a Meta object if not exists
        if 'Meta' not in kls.__dict__:
            kls.Meta = type('Meta', (object,), {})
        if 'unique_together' not in kls.Meta.__dict__:
            kls.Meta.unique_together = []
        # set verbose_name(s) if not already set
        if 'verbose_name' not in kls.Meta.__dict__:
            kls.Meta.verbose_name = kls.__name__
        if 'verbose_name_plural' not in kls.Meta.__dict__:
            kls.Meta.verbose_name_plural = kls.Meta.verbose_name + 's'