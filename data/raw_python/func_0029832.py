def accept(self):
        '''
        Acts as `Field.accepts` but returns result of every child field 
        as value in parent `python_data`.
        '''
        result = FieldSet.accept(self)
        self.clean_value = result[self.name]
        return self.clean_value