def get(self, labels):
        """ Get gets the data in the form of 0.5, 0.9 and 0.99 percentiles. Also
            you get sum and count, all in a dict
        """

        return_data = {}

        # We have already a lock for data but not for the estimator
        with mutex:
            e = self.get_value(labels)

            # Set invariants data (default to 0.50, 0.90 and 0.99)
            for i in e._invariants:
                q = i._quantile
                return_data[q] = e.query(q)

            # Set sum and count
            return_data[self.__class__.SUM_KEY] = e._sum
            return_data[self.__class__.COUNT_KEY] = e._observations

        return return_data