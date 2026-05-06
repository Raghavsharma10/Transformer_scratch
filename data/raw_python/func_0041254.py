def update(self, attrs):
        """update attributes initialized with the proper type"""
        self._validateAttrs(attrs)
        for k,v in attrs.items():
            typecast = type( getattr(self, k) )
            if typecast==bool and v=="False":   newval = False # "False" evalued as boolean is True because its length > 0
            else:                               newval = typecast(v.lower())
            setattr(self, k, newval)