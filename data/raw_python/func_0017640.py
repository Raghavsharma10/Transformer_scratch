def values(self):
        """
        Returns a list of values for this field for this instance. It's a list
        so we can accomodate many-to-many fields.
        """
        # This import is deliberately inside the function because it causes
        # some settings to be imported, and we don't want to do that at the
        # module level.
        if self.field.rel:
            if isinstance(self.field.rel, models.ManyToOneRel):
                objs = getattr(self.instance.instance, self.field.name)
            elif isinstance(self.field.rel,
                            models.ManyToManyRel): # ManyToManyRel
                return list(getattr(self.instance.instance,
                                    self.field.name).all())
        elif self.field.choices:
            objs = dict(self.field.choices).get(self.raw_value, EMPTY_VALUE)
        elif isinstance(self.field, models.DateField) or \
                                    isinstance(self.field, models.TimeField):
            if self.raw_value:
                if isinstance(self.field, models.DateTimeField):
                    objs = capfirst(formats.date_format(self.raw_value,
                                                        'DATETIME_FORMAT'))
                elif isinstance(self.field, models.TimeField):
                    objs = capfirst(formats.time_format(self.raw_value,
                                                        'TIME_FORMAT'))
                else:
                    objs = capfirst(formats.date_format(self.raw_value,
                                                        'DATE_FORMAT'))
            else:
                objs = EMPTY_VALUE
        elif isinstance(self.field, models.BooleanField) or \
                            isinstance(self.field, models.NullBooleanField):
            objs = {True: 'Yes', False: 'No', None: 'Unknown'}[self.raw_value]
        else:
            objs = self.raw_value
        return [objs]