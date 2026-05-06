def Field(self, field, Value = None):
        '''
        Add field to bitmap
        '''
        if Value == None:
            try:
                return self.__Bitmap[field]
            except KeyError:
                return None
        elif Value == 1 or Value == 0:
            self.__Bitmap[field] = Value
        else:
            raise ValueError