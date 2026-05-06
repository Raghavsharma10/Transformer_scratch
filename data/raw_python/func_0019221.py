def effective_max_ar_order(self):
        """The maximum number of AR coefficients that shall or can be
        determined.

        It is the minimum of |ARMA.max_ar_order| and the number of
        coefficients of the pure |MA| after their turning point.
        """
        return min(self.max_ar_order, self.ma.order-self.ma.turningpoint[0]-1)