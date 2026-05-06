def list_wegsegmenten_by_straat(self, straat):
        '''
        List all `wegsegmenten` in a :class:`Straat`

        :param straat: The :class:`Straat` for which the `wegsegmenten` \
                are wanted.
        :rtype: A :class:`list` of :class:`Wegsegment`
        '''
        try:
            id = straat.id
        except AttributeError:
            id = straat

        def creator():
            res = crab_gateway_request(
                self.client, 'ListWegsegmentenByStraatnaamId', id
            )
            try:
                return[
                    Wegsegment(
                        r.IdentificatorWegsegment,
                        r.StatusWegsegment
                    )for r in res.WegsegmentItem
                ]
            except AttributeError:
                return []
        if self.caches['short'].is_configured:
            key = 'ListWegsegmentenByStraatnaamId#%s' % (id)
            wegsegmenten = self.caches['short'].get_or_create(key, creator)
        else:
            wegsegmenten = creator()
        for r in wegsegmenten:
            r.set_gateway(self)
        return wegsegmenten