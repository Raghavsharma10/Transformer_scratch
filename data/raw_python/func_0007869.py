def _terms(self):
        """ Returns a list with the objects as terms. """
        res = []
        for sign, terms in self.terms.items():
            for ID, lon in terms.items():
                res.append(self.T(ID, sign))
        return res