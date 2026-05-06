def intersection(self, another_moc, delta_t=DEFAULT_OBSERVATION_TIME):
        """
        Intersection between self and moc. ``delta_t`` gives the possibility to the user
        to set a time resolution for performing the tmoc intersection

        Parameters
        ----------
        another_moc : `~mocpy.abstract_moc.AbstractMOC`
            the MOC/TimeMOC used for performing the intersection with self
        delta_t : `~astropy.time.TimeDelta`, optional
            the duration of one observation. It is set to 30 min by default. This data is used to compute the
            more efficient TimeMoc order to represent the observations. (Best order = the less precise order which
            is able to discriminate two observations separated by ``delta_t``)

        Returns
        -------
        result : `~mocpy.moc.MOC` or `~mocpy.tmoc.TimeMOC`
            MOC object whose interval set corresponds to : self & ``moc``

        """

        order_op = TimeMOC.time_resolution_to_order(delta_t)

        self_degraded, moc_degraded = self._process_degradation(another_moc, order_op)
        return super(TimeMOC, self_degraded).intersection(moc_degraded)