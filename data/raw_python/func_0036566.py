def _detect_loop(self):
        """ detect loops in flow table, raise error if being present
        """
        for source, dests in self.flowtable.items():
            if source in dests:
                raise conferr('Loops detected: %s --> %s' % (source, source))