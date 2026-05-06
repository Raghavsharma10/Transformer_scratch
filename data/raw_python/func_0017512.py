def change_domain(self, domain):
        """
        Creating new Curve object in memory with domain passed as a parameter.
        New domain must include in the original domain.
        Copies values from original curve and uses interpolation to calculate
        values for new points in domain.

        Calculate y - values of example curve with changed domain:
        >>> print(Curve([[0,0], [5, 5], [10, 0]])\
            .change_domain([1, 2, 8, 9]).y)
        [1. 2. 2. 1.]

        :param domain: set of points representing new domain.
            Might be a list or np.array.
        :return: new Curve object with domain set by 'domain' parameter
        """
        logger.info('Running %(name)s.change_domain() with new domain range:[%(ymin)s, %(ymax)s]',
                    {"name": self.__class__, "ymin": np.min(domain), "ymax": np.max(domain)})

        # check if new domain includes in the original domain
        if np.max(domain) > np.max(self.x) or np.min(domain) < np.min(self.x):
            logger.error('Old domain range: [%(xmin)s, %(xmax)s] does not include new domain range:'
                         '[%(ymin)s, %(ymax)s]', {"xmin": np.min(self.x), "xmax": np.max(self.x),
                                                  "ymin": np.min(domain), "ymax": np.max(domain)})
            raise ValueError('in change_domain():' 'the old domain does not include the new one')

        y = np.interp(domain, self.x, self.y)
        # We need to join together domain and values (y) because we are recreating Curve object
        # (we pass it as argument to self.__class__)
        # np.dstack((arrays), axis=1) joins given arrays like np.dstack() but it also nests the result
        # in additional list and this is the reason why we use [0] to remove this extra layer of list like this:
        # np.dstack([[0, 5, 10], [0, 0, 0]]) gives [[[ 0,  0], [ 5,  0], [10,  0]]] so use dtack()[0]
        # to get this: [[0,0], [5, 5], [10, 0]]
        # which is a 2 dimensional array and can be used to create a new Curve object
        obj = self.__class__(np.dstack((domain, y))[0], **self.__dict__['metadata'])
        return obj