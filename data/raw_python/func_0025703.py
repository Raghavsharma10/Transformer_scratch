def sync(self, *sids):
        """
        Synchronise data

        Parameters
        ----------
        sids : list of str
            SensorIDs to sync
            Optional, leave empty to sync everything
        """
        if sids == ():
            sids = [sid for (sid,) in self.dbcur.execute(SQL_SENSOR_ALL)]
        for sid in sids:
            self.dbcur.execute(SQL_TMPO_LAST, (sid,))
            last = self.dbcur.fetchone()
            if last:
                rid, lvl, bid, ext = last
                self._clean(sid, rid, lvl, bid)
                # prevent needless polling
                if time.time() < bid + 256:
                    return
            else:
                rid, lvl, bid = 0, 0, 0
            self._req_sync(sid, rid, lvl, bid)