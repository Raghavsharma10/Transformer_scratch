def calculate_single_terms(self):
        """Lines of model method with the same name."""
        lines = self._call_methods('calculate_single_terms',
                                   self.model.PART_ODE_METHODS)
        if lines:
            lines.insert(1, ('        self.numvars.nmb_calls ='
                             'self.numvars.nmb_calls+1'))
        return lines