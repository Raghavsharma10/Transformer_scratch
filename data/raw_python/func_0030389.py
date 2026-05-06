def _sign_payload(self, payload):
        """使用 appkey 对 payload 进行签名，返回新的请求参数
        """
        app_key = self._app_key
        t = int(time.time() * 1000)
        requestStr = {
            'header': self._req_header,
            'model': payload
        }
        data = json.dumps({'requestStr': json.dumps(requestStr)})
        data_str = '{}&{}&{}&{}'.format(self._req_token, t, app_key, data)
        sign = hashlib.md5(data_str.encode('utf-8')).hexdigest()
        params = {
            't': t,
            'appKey': app_key,
            'sign': sign,
            'data': data,
        }
        return params