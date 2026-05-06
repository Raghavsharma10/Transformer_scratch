def fetch(self, value_obj=None):
        ''' Fetch the next two values '''
        val = None
        try:
            val = next(self.__iterable)
        except StopIteration:
            return None
        if value_obj is None:
            value_obj = Value(value=val)
        else:
            value_obj.value = val
        return value_obj