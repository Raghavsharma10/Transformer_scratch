def gaussian(cls, mu=0, sigma=1):
        '''
        :mu:     mean
        :sigma:  standard deviation
        :return: Point subclass

        Returns a point whose coordinates are picked from a Gaussian
        distribution with mean 'mu' and standard deviation 'sigma'.
        See random.gauss for further explanation of those parameters.
        '''
        return cls(random.gauss(mu, sigma),
                   random.gauss(mu, sigma),
                   random.gauss(mu, sigma))