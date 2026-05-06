def rebinned(self, step=0.1, fixp=0):
        """
        Provides effective way to compute new domain basing on
        step and fixp parameters. Then using change_domain() method
        to create new object with calculated domain and returns it.

        fixp doesn't have to be inside original domain.

        Return domain of a new curve specified by
        fixp=0 and step=1 and another Curve object:
        >>> print(Curve([[0,0], [5, 5], [10, 0]]).rebinned(1, 0).x)
        [  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]

        :param step: step size of new domain
        :param fixp: fixed point one of the points in new domain
        :return: new Curve object with domain specified by
            step and fixp parameters
        """
        logger.info('Running %(name)s.rebinned(step=%(st)s, fixp=%(fx)s)',
                    {"name": self.__class__, "st": step, "fx": fixp})
        a, b = (np.min(self.x), np.max(self.x))
        count_start = abs(fixp - a) / step
        count_stop = abs(fixp - b) / step

        # depending on position of fixp with respect to the original domain
        # 3 cases may occur:
        if fixp < a:
            count_start = math.ceil(count_start)
            count_stop = math.floor(count_stop)
        elif fixp > b:
            count_start = -math.floor(count_start)
            count_stop = -math.ceil(count_stop)
        else:
            count_start = -count_start
            count_stop = count_stop

        domain = [fixp + n * step for n in range(int(count_start), int(count_stop) + 1)]
        return self.change_domain(domain)