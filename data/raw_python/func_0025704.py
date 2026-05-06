def list(self, *sids):
        """
        List all tmpo-blocks in the database

        Parameters
        ----------
        sids : list of str
            SensorID's for which to list blocks
            Optional, leave empty to get them all

        Returns
        -------
        list[list[tuple]]
        """
        if sids == ():
            sids = [sid for (sid,) in self.dbcur.execute(SQL_SENSOR_ALL)]
        slist = []
        for sid in sids:
            tlist = []
            for tmpo in self.dbcur.execute(SQL_TMPO_ALL, (sid,)):
                tlist.append(tmpo)
                sid, rid, lvl, bid, ext, ctd, blk = tmpo
                self._dprintf(DBG_TMPO_WRITE, ctd, sid, rid, lvl, bid, len(blk))
            slist.append(tlist)
        return slist