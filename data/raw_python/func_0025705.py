def series(self, sid, recycle_id=None, head=None, tail=None,
               datetime=True):
        """
        Create data Series

        Parameters
        ----------
        sid : str
        recycle_id : optional
        head : int | pandas.Timestamp, optional
            Start of the interval
            default earliest available
        tail : int | pandas.Timestamp, optional
            End of the interval
            default max epoch
        datetime : bool
            convert index to datetime
            default True

        Returns
        -------
        pandas.Series
        """
        if head is None:
            head = 0
        else:
            head = self._2epochs(head)

        if tail is None:
            tail = EPOCHS_MAX
        else:
            tail = self._2epochs(tail)

        if recycle_id is None:
            self.dbcur.execute(SQL_TMPO_RID_MAX, (sid,))
            recycle_id = self.dbcur.fetchone()[0]
        tlist = self.list(sid)[0]
        srlist = []
        for _sid, rid, lvl, bid, ext, ctd, blk in tlist:
            if (recycle_id == rid
            and head < self._blocktail(lvl, bid)
            and tail >= bid):
                srlist.append(self._blk2series(ext, blk, head, tail))
        if len(srlist) > 0:
            ts = pd.concat(srlist)
            ts.name = sid
            if datetime is True:
                ts.index = pd.to_datetime(ts.index, unit="s", utc=True)
            return ts
        else:
            return pd.Series([], name=sid)