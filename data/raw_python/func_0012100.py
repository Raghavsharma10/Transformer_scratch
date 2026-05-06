def call(self, callname, data=None, **args):
        """
        Generic interface to REST apiGeneric interface to REST api
        :param callname:  query name
        :param data:   dictionary of inputs
        :param args:    keyword arguments added to the payload
        :return:
        """
        url = f"{self.url_base}/{callname}"
        payload = self.payload.copy()
        payload.update(**args)

        if data is not None:
            payload.update(data)

        res = self.session.post(url, data=payload)

        if res.status_code > 299:
            self.log.error(f"URL: {url}")
            self.log.error(f"Payload: {payload}")
            self.log.error(f"STATUS: {res.status_code}")
            self.log.error(f"RESPONSE: {res.text}")
            return
        elif 'error' in res.json():
            self.log.error(res.json()['error'])
            return

        return res.json()