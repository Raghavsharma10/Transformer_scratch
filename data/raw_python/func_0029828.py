def clean_value(self):
        '''
        Current field's converted value from form's python_data.
        '''
        # XXX cached_property is used only for set initial state
        #     this property should be set every time field data
        #     has been changed, for instance, in accept method
        python_data = self.parent.python_data
        if self.name in python_data:
            return python_data[self.name]
        return self.get_initial()