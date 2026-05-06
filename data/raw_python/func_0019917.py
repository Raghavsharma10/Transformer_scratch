def prefixes_for_asn(self, asn):
        '''Gets the AS information for a given ASN. Return the CIDR and geolocation associated with the AS.'''

        uri = self._uris["prefixes_for_asn"].format(asn)
        resp_json = self.get_parse(uri)

        return resp_json