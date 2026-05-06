def _get_data_raw(self, time, site_id):
        """Download data from the University of Wyoming's upper air archive.

        Parameters
        ----------
        time : datetime
            Date and time for which data should be downloaded
        site_id : str
            Site id for which data should be downloaded

        Returns
        -------
        text of the server response

        """
        path = ('?region=naconf&TYPE=TEXT%3ALIST'
                '&YEAR={time:%Y}&MONTH={time:%m}&FROM={time:%d%H}&TO={time:%d%H}'
                '&STNM={stid}').format(time=time, stid=site_id)

        resp = self.get_path(path)
        # See if the return is valid, but has no data
        if resp.text.find('Can\'t') != -1:
            raise ValueError(
                'No data available for {time:%Y-%m-%d %HZ} '
                'for station {stid}.'.format(time=time, stid=site_id))

        return resp.text