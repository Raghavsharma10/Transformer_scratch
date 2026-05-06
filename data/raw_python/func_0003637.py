def match(elem, seq_expr):
        """
        Return True if elem (an element of elem_list) matches seq_expr, an element in self.sequence
        """
        if type(seq_expr) is str:  # wild-card
            if seq_expr == '.':  # match any element
                return True
            elif seq_expr == '\d':
                return elem.is_numerical()
            elif seq_expr == '\D':
                return not elem.is_numerical()
            else:  # invalid wild-card specified
                raise LookupError('{0} is not a valid wild-card'.format(seq_expr))
        else:  # date element
            return elem == seq_expr