def accept(self, context):
        """ Check if the context could be accepted by the oracle
        
        Args:
            context: s sequence same type as the oracle data
        
        Returns:
            bAccepted: whether the sequence is accepted or not
            _next: the state where the sequence is accepted
        """
        _next = 0
        for _s in context:
            _data = [self.data[j] for j in self.trn[_next]]
            if _s in _data:
                _next = self.trn[_next][_data.index(_s)]
            else:
                return 0, _next
        return 1, _next