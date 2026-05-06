def _lookup_qrz_callsign(self, callsign=None, apikey=None, apiv="1.3.3"):
        """ Performs the callsign lookup against the QRZ.com XML API:
        """

        if apikey is None:
            raise AttributeError("Session Key Missing")

        callsign = callsign.upper()

        response = self._request_callsign_info_from_qrz(callsign, apikey, apiv)

        root = BeautifulSoup(response.text, "html.parser")
        lookup = {}

        if root.error:

            if re.search('Not found', root.error.text, re.I):  #No data available for callsign
                raise KeyError(root.error.text)

            #try to get a new session key and try to request again
            elif re.search('Session Timeout', root.error.text, re.I) or re.search('Invalid session key', root.error.text, re.I):
                apikey = self._get_qrz_session_key(self._username, self._pwd)
                response = self._request_callsign_info_from_qrz(callsign, apikey, apiv)
                root = BeautifulSoup(response.text, "html.parser")

                #if this fails again, raise error
                if root.error:

                    if re.search('Not found', root.error.text, re.I):  #No data available for callsign
                        raise KeyError(root.error.text)
                    else:
                        raise AttributeError(root.error.text) #most likely session key invalid
                else:
                    #update API Key ob Lookup object
                    self._apikey = apikey

            else:
                raise AttributeError(root.error.text) #most likely session key missing

        if root.callsign is None:
            raise ValueError

        if root.callsign.call:
            lookup[const.CALLSIGN] = root.callsign.call.text
        if root.callsign.xref:
            lookup[const.XREF] = root.callsign.xref.text
        if root.callsign.aliases:
            lookup[const.ALIASES] = root.callsign.aliases.text.split(',')
        if root.callsign.dxcc:
            lookup[const.ADIF] = int(root.callsign.dxcc.text)
        if root.callsign.fname:
            lookup[const.FNAME] = root.callsign.fname.text
        if root.callsign.find("name"):
            lookup[const.NAME] = root.callsign.find('name').get_text()
        if root.callsign.addr1:
            lookup[const.ADDR1] = root.callsign.addr1.text
        if root.callsign.addr2:
            lookup[const.ADDR2] = root.callsign.addr2.text
        if root.callsign.state:
            lookup[const.STATE] = root.callsign.state.text
        if root.callsign.zip:
            lookup[const.ZIPCODE] = root.callsign.zip.text
        if root.callsign.country:
            lookup[const.COUNTRY] = root.callsign.country.text
        if root.callsign.ccode:
            lookup[const.CCODE] = int(root.callsign.ccode.text)
        if root.callsign.lat:
            lookup[const.LATITUDE] = float(root.callsign.lat.text)
        if root.callsign.lon:
            lookup[const.LONGITUDE] = float(root.callsign.lon.text)
        if root.callsign.grid:
            lookup[const.LOCATOR] = root.callsign.grid.text
        if root.callsign.county:
            lookup[const.COUNTY] = root.callsign.county.text
        if root.callsign.fips:
            lookup[const.FIPS] = int(root.callsign.fips.text) # check type
        if root.callsign.land:
            lookup[const.LAND] = root.callsign.land.text
        if root.callsign.efdate:
            try:
                lookup[const.EFDATE] = datetime.strptime(root.callsign.efdate.text, '%Y-%m-%d').replace(tzinfo=UTC)
            except ValueError:
                self._logger.debug("[QRZ.com] efdate: Invalid DateTime; " + callsign + " " + root.callsign.efdate.text)
        if root.callsign.expdate:
            try:
                lookup[const.EXPDATE] = datetime.strptime(root.callsign.expdate.text, '%Y-%m-%d').replace(tzinfo=UTC)
            except ValueError:
                self._logger.debug("[QRZ.com] expdate: Invalid DateTime; " + callsign + " " + root.callsign.expdate.text)
        if root.callsign.p_call:
            lookup[const.P_CALL] = root.callsign.p_call.text
        if root.callsign.find('class'):
             lookup[const.LICENSE_CLASS] = root.callsign.find('class').get_text()
        if root.callsign.codes:
            lookup[const.CODES] = root.callsign.codes.text
        if root.callsign.qslmgr:
            lookup[const.QSLMGR] = root.callsign.qslmgr.text
        if root.callsign.email:
            lookup[const.EMAIL] = root.callsign.email.text
        if root.callsign.url:
            lookup[const.URL] = root.callsign.url.text
        if root.callsign.u_views:
            lookup[const.U_VIEWS] = int(root.callsign.u_views.text)
        if root.callsign.bio:
            lookup[const.BIO] = root.callsign.bio.text
        if root.callsign.biodate:
            try:
                lookup[const.BIODATE] = datetime.strptime(root.callsign.biodate.text, '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
            except ValueError:
                self._logger.warning("[QRZ.com] biodate: Invalid DateTime; " + callsign)
        if root.callsign.image:
            lookup[const.IMAGE] = root.callsign.image.text
        if root.callsign.imageinfo:
            lookup[const.IMAGE_INFO] = root.callsign.imageinfo.text
        if root.callsign.serial:
            lookup[const.SERIAL] = long(root.callsign.serial.text)
        if root.callsign.moddate:
            try:
                lookup[const.MODDATE] = datetime.strptime(root.callsign.moddate.text, '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
            except ValueError:
                self._logger.warning("[QRZ.com] moddate: Invalid DateTime; " + callsign)
        if root.callsign.MSA:
            lookup[const.MSA] = int(root.callsign.MSA.text)
        if root.callsign.AreaCode:
            lookup[const.AREACODE] = int(root.callsign.AreaCode.text)
        if root.callsign.TimeZone:
            lookup[const.TIMEZONE] = int(root.callsign.TimeZone.text)
        if root.callsign.GMTOffset:
            lookup[const.GMTOFFSET] = float(root.callsign.GMTOffset.text)
        if root.callsign.DST:
            if root.callsign.DST.text == "Y":
                lookup[const.DST] = True
            else:
                lookup[const.DST] = False
        if root.callsign.eqsl:
            if root.callsign.eqsl.text == "1":
                lookup[const.EQSL] = True
            else:
                lookup[const.EQSL] = False
        if root.callsign.mqsl:
            if root.callsign.mqsl.text == "1":
                lookup[const.MQSL] = True
            else:
                lookup[const.MQSL] = False
        if root.callsign.cqzone:
            lookup[const.CQZ] = int(root.callsign.cqzone.text)
        if root.callsign.ituzone:
            lookup[const.ITUZ] = int(root.callsign.ituzone.text)
        if root.callsign.born:
            lookup[const.BORN] = int(root.callsign.born.text)
        if root.callsign.user:
            lookup[const.USER_MGR] = root.callsign.user.text
        if root.callsign.lotw:
            if root.callsign.lotw.text == "1":
                lookup[const.LOTW] = True
            else:
                lookup[const.LOTW] = False
        if root.callsign.iota:
            lookup[const.IOTA] = root.callsign.iota.text
        if root.callsign.geoloc:
            lookup[const.GEOLOC] = root.callsign.geoloc.text

        # if sys.version_info >= (2,):
        #     for item in lookup:
        #         if isinstance(lookup[item], unicode):
        #             print item, repr(lookup[item])
        return lookup