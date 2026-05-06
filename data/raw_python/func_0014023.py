def clean(self):
        """
        Make sure the lookup makes sense
        """
        if self.lookup == '?':  # Randomly sort
            return
        else:
            lookups = self.lookup.split(LOOKUP_SEP)
            opts = self.model_def.model_class()._meta
            valid = True
            while len(lookups):
                lookup = lookups.pop(0)
                try:
                    field = opts.get_field(lookup)
                except FieldDoesNotExist:
                    valid = False
                else:
                    if isinstance(field, models.ForeignKey):
                        opts = get_remote_field_model(field)._meta
                    elif len(lookups):  # Cannot go any deeper
                        valid = False
                finally:
                    if not valid:
                        msg = _("This field doesn't exist")
                        raise ValidationError({'lookup': [msg]})