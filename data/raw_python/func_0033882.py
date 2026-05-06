def get_project_members(self):
        """Return the list of members in the project"""

        r = self._request('members/')
        if not r:
            return None

        retour = []

        for data in r.json()['members']:
            # Base properties
            u = User()
            u.__dict__.update(data)

            retour.append(u)

        return retour