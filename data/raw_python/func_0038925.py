def search(self, lat_range=None, long_range=None, variance=None,
               bssid=None, ssid=None,
               last_update=None,
               address=None, state=None, zipcode=None,
               on_new_page=None, max_results=100):
        """
        Search the Wigle wifi database for matching entries. The following
        criteria are supported:

        Args:
            lat_range ((float, float)): latitude range
            long_range ((float, float)): longitude range
            variance (float): radius tolerance in degrees
            bssid (str): BSSID/MAC of AP
            ssid (str): SSID of network
            last_update (datetime): when was the AP last seen
            address (str): location, address
            state (str): location, state
            zipcode (str): location, zip code
            on_new_page (func(int)): callback to notify when requesting new
                page of results

        Returns:
            [dict]: list of dicts describing matching wifis
        """

        params = {
            'latrange1': lat_range[0] if lat_range else "",
            'latrange2': lat_range[1] if lat_range else "",
            'longrange1': long_range[0] if long_range else "",
            'longrange2': long_range[1] if long_range else "",
            'variance': str(variance) if variance else "0.01",
            'netid': bssid or "",
            'ssid': ssid or "",
            'lastupdt': last_update.strftime("%Y%m%d%H%M%S") if last_update else "",
            'addresscode': address or "",
            'statecode': state or "",
            'zipcode': zipcode or "",
            'Query': "Query",
        }

        wifis = []
        while True:
            if on_new_page:
                on_new_page(params.get('first', 1))
            resp = self._authenticated_request('jsonSearch', params=params)
            data = resp.json()
            if not data['success']:
                raise_wigle_error(data)

            for result in data['results'][:max_results-len(wifis)]:
                normalise_entry(result)
                wifis.append(result)

            if data['resultCount'] < WIGLE_PAGESIZE or len(wifis) >= max_results:
                break

            params['first'] = data['last'] + 1

        return wifis