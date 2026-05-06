def auth_access(self, auth_code):
        """
        verify the fist authorization response url code

        response data
        返回值字段      字段类型    字段说明
        access_token    string      用户授权的唯一票据，用于调用微博的开放接口，同时也是第三方应用验证微博用户登录的唯一票据，
                                    第三方应用应该用该票据和自己应用内的用户建立唯一影射关系，来识别登录状态，不能使用本返回值里的UID
                                    字段来做登录识别。

        expires_in      string      access_token的生命周期，单位是秒数。
        remind_in       string      access_token的生命周期（该参数即将废弃，开发者请使用expires_in）。
        uid             string      授权用户的UID，本字段只是为了方便开发者，减少一次user/show接口调用而返回的，第三方应用不能用此字段作为用户
                                    登录状态的识别，只有access_token才是用户授权的唯一票据。

        :param auth_code: authorize_url response code
        :return:

        normal:
         {
               "access_token": "ACCESS_TOKEN",
               "expires_in": 1234,
               "remind_in":"798114",
               "uid":"12341234"
         }
         mobile:
         {
            "access_token": "SlAV32hkKG",
            "remind_in": 3600,
            "expires_in": 3600
            "refresh_token": "QXBK19xm62"
        }

        """
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'authorization_code',
            'code': auth_code,
            'redirect_uri': self.redirect_url
        }
        return self.request("post", "access_token", data=data)