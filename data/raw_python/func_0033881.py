def get_orga(self, orgaPk):
        """Return an organization speficied with orgaPk"""
        r = self._request('orga/' + str(orgaPk))
        if r:
            # Set base properties and copy data inside the orga
            o = Orga()
            o.pk = o.id = orgaPk
            o.__dict__.update(r.json())
            return o
        return None