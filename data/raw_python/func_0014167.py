def post(self, url, data, charset=CHARSET_UTF8, headers={}):
        '''response json text'''
        if 'Api-Lang' not in headers:
            headers['Api-Lang'] = 'python'
        if 'Content-Type' not in headers:
            headers['Content-Type'] = "application/x-www-form-urlencoded;charset=" + charset
        rsp = requests.post(url, data, headers=headers,
                            timeout=(int(self.conf(HTTP_CONN_TIMEOUT, '10')), int(self.conf(HTTP_SO_TIMEOUT, '30'))))
        return json.loads(rsp.text)