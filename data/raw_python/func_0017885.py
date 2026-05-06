def decode_metar(self, metar):
        """
        Simple method that decodes a given metar string.

        Args:
            metar (str): The metar data

        Returns:
            The metar data in readable format

        Example::

            from pyflightdata import FlightData
            f=FlightData()
            f.decode_metar('WSSS 181030Z 04009KT 010V080 9999 FEW018TCU BKN300 29/22 Q1007 NOSIG')
        """
        try:
            from metar import Metar
        except:
            return "Unable to parse metars. Please install parser from https://github.com/tomp/python-metar."
        m = Metar.Metar(metar)
        return m.string()