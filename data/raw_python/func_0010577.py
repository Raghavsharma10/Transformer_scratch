def _dismantle_callsign(self, callsign, timestamp=timestamp_now):
        """ try to identify the callsign's identity by analyzing it in the following order:

        Args:
            callsign (str): Amateur Radio callsign
            timestamp (datetime, optional): datetime in UTC (tzinfo=pytz.UTC)

        Raises:
            KeyError: Callsign could not be identified


        """
        entire_callsign = callsign.upper()

        if re.search('[/A-Z0-9\-]{3,15}', entire_callsign):  # make sure the call has at least 3 characters

            if re.search('\-\d{1,3}$', entire_callsign):  # cut off any -10 / -02 appendixes
                callsign = re.sub('\-\d{1,3}$', '', entire_callsign)

            if re.search('/[A-Z0-9]{1,4}/[A-Z0-9]{1,4}$', callsign):
                callsign = re.sub('/[A-Z0-9]{1,4}$', '', callsign)  # cut off 2. appendix DH1TW/HC2/P -> DH1TW/HC2

            # multiple character appendix (callsign/xxx)
            if re.search('[A-Z0-9]{4,10}/[A-Z0-9]{2,4}$', callsign):  # case call/xxx, but ignoring /p and /m or /5
                appendix = re.search('/[A-Z0-9]{2,4}$', callsign)
                appendix = re.sub('/', '', appendix.group(0))
                self._logger.debug("appendix: " + appendix)

                if appendix == 'MM':  # special case Martime Mobile
                    #self._mm = True
                    return {
                        'adif': 999,
                        'continent': '',
                        'country': 'MARITIME MOBILE',
                        'cqz': 0,
                        'latitude': 0.0,
                        'longitude': 0.0
                    }
                elif appendix == 'AM':  # special case Aeronautic Mobile
                    return {
                        'adif': 998,
                        'continent': '',
                        'country': 'AIRCAFT MOBILE',
                        'cqz': 0,
                        'latitude': 0.0,
                        'longitude': 0.0
                    }
                elif appendix == 'QRP':  # special case QRP
                    callsign = re.sub('/QRP', '', callsign)
                    return self._iterate_prefix(callsign, timestamp)
                elif appendix == 'QRPP':  # special case QRPP
                    callsign = re.sub('/QRPP', '', callsign)
                    return self._iterate_prefix(callsign, timestamp)
                elif appendix == 'BCN':  # filter all beacons
                    callsign = re.sub('/BCN', '', callsign)
                    data = self._iterate_prefix(callsign, timestamp).copy()
                    data[const.BEACON] = True
                    return data
                elif appendix == "LH":  # Filter all Lighthouses
                    callsign = re.sub('/LH', '', callsign)
                    return self._iterate_prefix(callsign, timestamp)
                elif re.search('[A-Z]{3}', appendix): #case of US county(?) contest N3HBX/UAL
                    callsign = re.sub('/[A-Z]{3}$', '', callsign)
                    return self._iterate_prefix(callsign, timestamp)

                else:
                    # check if the appendix is a valid country prefix
                    return self._iterate_prefix(re.sub('/', '', appendix), timestamp)

            # Single character appendix (callsign/x)
            elif re.search('/[A-Z0-9]$', callsign):  # case call/p or /b /m or /5 etc.
                appendix = re.search('/[A-Z0-9]$', callsign)
                appendix = re.sub('/', '', appendix.group(0))

                if appendix == 'B':  # special case Beacon
                    callsign = re.sub('/B', '', callsign)
                    data = self._iterate_prefix(callsign, timestamp).copy()
                    data[const.BEACON] = True
                    return data

                elif re.search('\d$', appendix):
                    area_nr = re.search('\d$', appendix).group(0)
                    callsign = re.sub('/\d$', '', callsign) #remove /number
                    if len(re.findall(r'\d+', callsign)) == 1: #call has just on digit e.g. DH1TW
                        callsign = re.sub('[\d]+', area_nr, callsign)
                    else: # call has several digits e.g. 7N4AAL
                        pass # no (two) digit prefix contries known where appendix would change entitiy
                    return self._iterate_prefix(callsign, timestamp)

                else:
                    return self._iterate_prefix(callsign, timestamp)

            # regular callsigns, without prefix or appendix
            elif re.match('^[\d]{0,1}[A-Z]{1,2}\d([A-Z]{1,4}|\d{3,3}|\d{1,3}[A-Z])[A-Z]{0,5}$', callsign):
                return self._iterate_prefix(callsign, timestamp)

            # callsigns with prefixes (xxx/callsign)
            elif re.search('^[A-Z0-9]{1,4}/', entire_callsign):
                pfx = re.search('^[A-Z0-9]{1,4}/', entire_callsign)
                pfx = re.sub('/', '', pfx.group(0))
                #make sure that the remaining part is actually a callsign (avoid: OZ/JO81)
                rest = re.search('/[A-Z0-9]+', entire_callsign)
                rest = re.sub('/', '', rest.group(0))
                if re.match('^[\d]{0,1}[A-Z]{1,2}\d([A-Z]{1,4}|\d{3,3}|\d{1,3}[A-Z])[A-Z]{0,5}$', rest):
                    return self._iterate_prefix(pfx)

        if entire_callsign in callsign_exceptions:
            return self._iterate_prefix(callsign_exceptions[entire_callsign])

        self._logger.debug("Could not decode " + callsign)
        raise KeyError("Callsign could not be decoded")