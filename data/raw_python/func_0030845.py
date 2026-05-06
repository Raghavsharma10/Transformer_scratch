def prepare_data(self):
        '''
        Method returning data passed to template.
        Subclasses can override it.
        '''
        value = self.get_raw_value()
        return dict(widget=self,
                    field=self.field,
                    value=value,
                    readonly=not self.field.writable)