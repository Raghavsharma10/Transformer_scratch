def label(self, name, color, update=True):
        """Create or update a label
        """
        url = '%s/labels' % self
        data = dict(name=name, color=color)
        response = self.http.post(
            url, json=data, auth=self.auth, headers=self.headers
        )
        if response.status_code == 201:
            return True
        elif response.status_code == 422 and update:
            url = '%s/%s' % (url, name)
            response = self.http.patch(
                url, json=data, auth=self.auth, headers=self.headers
            )
        response.raise_for_status()
        return False