def dataframe(self, sids, head=0, tail=EPOCHS_MAX, datetime=True):
        """
        Create data frame

        Parameters
        ----------
        sids : list[str]
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
        pandas.DataFrame
        """
        if head is None:
            head = 0
        else:
            head = self._2epochs(head)

        if tail is None:
            tail = EPOCHS_MAX
        else:
            tail = self._2epochs(tail)

        series = [self.series(sid, head=head, tail=tail, datetime=False)
                  for sid in sids]
        df = pd.concat(series, axis=1)
        if datetime is True:
            df.index = pd.to_datetime(df.index, unit="s", utc=True)
        return df