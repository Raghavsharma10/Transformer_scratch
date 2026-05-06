def localize(self, *positions, **kwargs):
        """
        convert global coordinate(s) to local taking
        introns into account and cds/tx-Start depending on cdna=True kwarg
        """
        cdna = kwargs.get('cdna', False)
        # TODO: account for strand ?? add kwarg ??
        # if it's to the CDNA, then it's based on the cdsStart
        start, end = (self.cdsStart, self.cdsEnd) if cdna else \
                                        (self.start, self.end)
        introns = self.introns or None
        if cdna:
            if not self.is_coding:
                return ([None] * len(positions)) if len(positions) > 1 else None
            introns = self._introns(self.cds) or None

        if introns is None:
            local_ps = [p - start if (start <= p < end) else None for p in positions]
            return local_ps[0] if len(positions) == 1 else local_ps

        introns = [(s - start, e - start) for s, e in introns]
        positions = [p - start for p in positions]
        # now both introns and positions are local starts based on cds/tx-Start
        local_ps = []
        l = end - start
        for original_p in positions:
            subtract = 0
            p = original_p
            print(p, l, file=sys.stderr)
            if p < 0 or p >= l: # outside of transcript
                local_ps.append(None)
                continue
            for s, e in introns:
                # within intron
                if s <= p <= e:
                    subtract = None
                    break
                # otherwise, adjust for intron length.
                elif p >= e:
                    subtract += (e - s)

            local_ps.append(p - subtract if subtract is not None else None)

        assert all(p is None or p >=0 for p in local_ps), (local_ps)
        return local_ps[0] if len(positions) == 1 else local_ps