def add(self, rule: ControlRule = None, *, supply: float):
        """
        Register a new rule above a given ``supply`` threshold

        Registration supports a single-argument form for use as a decorator,
        as well as a two-argument form for direct application.
        Use the former for ``def`` or ``class`` definitions,
        and the later for ``lambda`` functions and existing callables.

        .. code:: python

            @control.add(supply=10)
            def linear(pool, interval):
                if pool.utilisation < 0.75:
                    return pool.supply - interval
                elif pool.allocation > 0.95:
                    return pool.supply + interval

            control.add(
                lambda pool, interval: pool.supply * (1.2 if pool.allocation > 0.75 else 0.9),
                supply=100
            )
        """
        if supply in self._thresholds:
            raise ValueError('rule for threshold %s re-defined' % supply)
        if rule is not None:
            self.rules.append((supply, rule))
            self._thresholds.add(supply)
            return rule
        else:
            return partial(self.add, supply=supply)