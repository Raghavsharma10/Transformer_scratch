def get_orgas(self):
        """Return the list of pk for all orgas"""

        r = self._request('orgas/')
        if not r:
            return None

        retour = []

        for data in r.json()['data']:
            o = Orga()
            o.__dict__.update(data)
            o.pk = o.id

            retour.append(o)

        return retour