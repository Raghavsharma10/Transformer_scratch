def get_seq_seq(self,obj,default=None):
        """Return sequence of sequences."""
        if is_sequence2(obj):
            return [self.get_seq(o,default) for o in obj]
        else:
            return [self.get_seq(obj,default)]