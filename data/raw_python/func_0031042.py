def _make_session():
        """Create session object.

        :rtype: requests.Session
        """

        sess = requests.Session()
        sess.mount('http://', requests.adapters.HTTPAdapter(max_retries=False))
        sess.mount('https://', requests.adapters.HTTPAdapter(max_retries=False))

        return sess