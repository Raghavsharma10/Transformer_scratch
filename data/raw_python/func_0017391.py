def getalgo(self, operation, name):
        '''Return the algorithm for *operation* named *name*'''
        if operation not in self._algorithms:
            raise NotAvailable('{0} not registered.'.format(operation))
        oper = self._algorithms[operation]
        try:
            return oper[name]
        except KeyError:
            raise NotAvailable('{0} algorithm {1} not registered.'
                               .format(operation, name))