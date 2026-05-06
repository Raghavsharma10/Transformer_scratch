def data(self):
        """
        Returns a dictionnary containing all the passed data and an item
        ``error_list`` which holds the result of :attr:`error_list`.
        """
        res = {'error_list': self.error_list}
        res.update(super(ValidationErrors, self).data)
        return res