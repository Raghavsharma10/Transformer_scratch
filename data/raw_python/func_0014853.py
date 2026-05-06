def get_seq(self,obj,default=None):
        """Return sequence."""
        if is_sequence(obj):
            return obj
        if is_number(obj): return [obj]
        if obj is None and default is not None:
            log.warning('using default value (%s)'%(default))
            return self.get_seq(default)
        raise ValueError('expected sequence|number but got %s'%(type(obj)))