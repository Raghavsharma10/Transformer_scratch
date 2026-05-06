def _process_degradation(self, another_moc, order_op):
        """
        Degrade (down-sampling) self and ``another_moc`` to ``order_op`` order

        Parameters
        ----------
        another_moc : `~mocpy.tmoc.TimeMoc`
        order_op : int
            the order in which self and ``another_moc`` will be down-sampled to.

        Returns
        -------
        result : (`~mocpy.tmoc.TimeMoc`, `~mocpy.tmoc.TimeMoc`)
            self and ``another_moc`` degraded TimeMocs

        """
        max_order = max(self.max_order, another_moc.max_order)
        if order_op > max_order:
            message = 'Requested time resolution for the operation cannot be applied.\n' \
                      'The TimeMoc object resulting from the operation is of time resolution {0} sec.'.format(
                TimeMOC.order_to_time_resolution(max_order).sec)
            warnings.warn(message, UserWarning)

        self_degradation = self.degrade_to_order(order_op)
        another_moc_degradation = another_moc.degrade_to_order(order_op)

        result = self_degradation, another_moc_degradation
        return result