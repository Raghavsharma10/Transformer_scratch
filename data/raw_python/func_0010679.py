def _lookup_qrz_dxcc(self, dxcc_or_callsign, apikey, apiv="1.3.3"):
        """ Performs the dxcc lookup against the QRZ.com XML API:
        """

        response = self._request_dxcc_info_from_qrz(dxcc_or_callsign, apikey, apiv=apiv)

        root = BeautifulSoup(response.text, "html.parser")
        lookup = {}

        if root.error: #try to get a new session key and try to request again

            if re.search('No DXCC Information for', root.error.text, re.I):  #No data available for callsign
                raise KeyError(root.error.text)
            elif re.search('Session Timeout', root.error.text, re.I): # Get new session key
                self._apikey = apikey = self._get_qrz_session_key(self._username, self._pwd)
                response = self._request_dxcc_info_from_qrz(dxcc_or_callsign, apikey)
                root = BeautifulSoup(response.text, "html.parser")
            else:
                raise AttributeError("Session Key Missing") #most likely session key missing or invalid

        if root.dxcc is None:
            raise ValueError

        if root.dxcc.dxcc:
            lookup[const.ADIF] = int(root.dxcc.dxcc.text)
        if root.dxcc.cc:
            lookup['cc'] = root.dxcc.cc.text
        if root.dxcc.cc:
            lookup['ccc'] = root.dxcc.ccc.text
        if root.find('name'):
            lookup[const.COUNTRY] = root.find('name').get_text()
        if root.dxcc.continent:
            lookup[const.CONTINENT] = root.dxcc.continent.text
        if root.dxcc.ituzone:
            lookup[const.ITUZ] = int(root.dxcc.ituzone.text)
        if root.dxcc.cqzone:
            lookup[const.CQZ] = int(root.dxcc.cqzone.text)
        if root.dxcc.timezone:
            lookup['timezone'] = float(root.dxcc.timezone.text)
        if root.dxcc.lat:
            lookup[const.LATITUDE] = float(root.dxcc.lat.text)
        if root.dxcc.lon:
            lookup[const.LONGITUDE] = float(root.dxcc.lon.text)

        return lookup