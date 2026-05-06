def zen(self):
        """Returns a quote from the Zen of GitHub. Yet another API Easter Egg

        :returns: str
        """
        url = self._build_url('zen')
        resp = self._get(url)
        return resp.content if resp.status_code == 200 else ''