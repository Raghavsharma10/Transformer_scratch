def get_homecall(callsign):
        """Strips off country prefixes (HC2/DH1TW) and activity suffixes (DH1TW/P).

        Args:
            callsign (str): Amateur Radio callsign

        Returns:
            str: callsign without country/activity pre/suffixes

        Raises:
            ValueError: No callsign found in string

        Example:
            The following code retrieves the home call for "HC2/DH1TW/P"

            >>> from pyhamtools import LookupLib, Callinfo
            >>> my_lookuplib = LookupLib(lookuptype="countryfile")
            >>> cic = Callinfo(my_lookuplib)
            >>> cic.get_homecall("HC2/DH1TW/P")
            DH1TW

        """

        callsign = callsign.upper()
        homecall = re.search('[\d]{0,1}[A-Z]{1,2}\d([A-Z]{1,4}|\d{3,3}|\d{1,3}[A-Z])[A-Z]{0,5}', callsign)
        if homecall:
            homecall = homecall.group(0)
            return homecall
        else:
            raise ValueError