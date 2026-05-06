def _iterate_prefix(self, callsign, timestamp=timestamp_now):
        """truncate call until it corresponds to a Prefix in the database"""
        prefix = callsign

        if re.search('(VK|AX|VI)9[A-Z]{3}', callsign): #special rule for VK9 calls
            if timestamp > datetime(2006,1,1, tzinfo=UTC):
                prefix = callsign[0:3]+callsign[4:5]

        while len(prefix) > 0:
            try:
                return self._lookuplib.lookup_prefix(prefix, timestamp)
            except KeyError:
                prefix = prefix.replace(' ', '')[:-1]
                continue
        raise KeyError