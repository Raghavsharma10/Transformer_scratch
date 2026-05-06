def FieldData(self, field, Value = None):
        '''
        Add field data
        '''
        if Value == None:
            try:
                return self.__FieldData[field]
            except KeyError:
                return None
        else:
            if len(str(Value)) > self.__IsoSpec.MaxLength(field):
                raise ValueError('Value length larger than field maximum ({0})'.format(self.__IsoSpec.MaxLength(field)))
            
            self.Field(field, Value=1)
            self.__FieldData[field] = Value