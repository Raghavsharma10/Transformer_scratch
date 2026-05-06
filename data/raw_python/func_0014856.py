def get_3_tuple_list(self,obj,default=None):
        """Return list of 3-tuples from
        sequence of a sequence,
        sequence - it is mapped to sequence of 3-sequences if possible
        number
        """
        if is_sequence2(obj):
            return [self.get_3_tuple(o,default) for o in obj]
        elif is_sequence(obj):
            return [self.get_3_tuple(obj[i:i+3],default) for i in range(0,len(obj),3)]
        else:
            return [self.get_3_tuple(obj,default)]