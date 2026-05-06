def promote(self, lane, svcs=None, meta=None):
        """promote a build so it is ready for an upper lane"""

        svcs, meta, lane = self._prep_for_release(lane, svcs=svcs, meta=meta)

        # iterate and mark as future release
        for svc in svcs:
            self.changes.append("Promoting: {}.release.future={}".format(svc, self.name))
            self.rcs.patch('service', svc, {
                "release": {"future": self.name}, # new way
                "statuses": {"future": time.time()},
            })

        return self