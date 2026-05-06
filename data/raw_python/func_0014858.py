def get_3_3_tuple_list(self,obj,default=None):
        """Return list of 3x3-tuples.
        """
        if is_sequence3(obj):
            return [self.get_3_3_tuple(o,default) for o in obj]
        return [self.get_3_3_tuple(obj,default)]