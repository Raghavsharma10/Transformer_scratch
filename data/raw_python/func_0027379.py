def authorize_url(self):
        """
        authorization url
        request weibo authorization url
        :return:
        code    string    用于第二步调用oauth2/access_token接口，获取授权后的access token。
        state    string    如果传递参数，会回传该参数
        """

        if self.oauth2_params and self.oauth2_params.get("display") == "mobile":
            auth_url = self.mobile_url + "authorize"
        else:
            auth_url = self.site_url + "authorize"

        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_url,
        }
        params.update(self.oauth2_params)

        params = filter_params(params)

        return "{auth_url}?{params}".format(auth_url=auth_url, params=urlencode(params))