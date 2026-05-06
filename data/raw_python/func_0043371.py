def list_adresposities_by_nummer_and_straat(self, nummer, straat):
        '''
        List all `adresposities` for a huisnummer and a :class:`Straat`.

        :param nummer: A string representing a certain huisnummer.
        :param straat: The :class:`Straat` for which the \
            `adresposities` are wanted. OR A straat id.
        :rtype: A :class:`list` of :class:`Adrespositie`
        '''
        try:
            sid = straat.id
        except AttributeError:
            sid = straat
        def creator():
            res = crab_gateway_request(
                self.client, 'ListAdrespositiesByHuisnummer', nummer, sid
            )
            try:
                return [Adrespositie(
                    r.AdrespositieId,
                    r.HerkomstAdrespositie
                )for r in res.AdrespositieItem]
            except AttributeError:
                return []
        if self.caches['short'].is_configured:
            key = 'ListAdrespositiesByHuisnummer#%s%s' % (nummer, sid)
            adresposities = self.caches['short'].get_or_create(key, creator)
        else:
            adresposities = creator()
        for a in adresposities:
            a.set_gateway(self)
        return adresposities