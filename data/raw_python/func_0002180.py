def _get_data_raw(self, time, site_id, pressure=None):
        r"""Download data from the Iowa State's upper air archive.

        Parameters
        ----------
        time : datetime
            Date and time for which data should be downloaded
        site_id : str
            Site id for which data should be downloaded
        pressure : float, optional
            Mandatory pressure level at which to request data (in hPa).

        Returns
        -------
        list of json data

        """
        query = {'ts': time.strftime('%Y%m%d%H00')}
        if site_id is not None:
            query['station'] = site_id
        if pressure is not None:
            query['pressure'] = pressure

        resp = self.get_path('raob.py', query)
        json_data = json.loads(resp.text)

        # See if the return is valid, but has no data
        if not (json_data['profiles'] and json_data['profiles'][0]['profile']):
            message = 'No data available '
            if time is not None:
                message += 'for {time:%Y-%m-%d %HZ} '.format(time=time)
            if site_id is not None:
                message += 'for station {stid}'.format(stid=site_id)
            if pressure is not None:
                message += 'for pressure {pres}'.format(pres=pressure)
            message = message[:-1] + '.'
            raise ValueError(message)
        return json_data