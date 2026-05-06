def user_get_session_token(self, app_id=None, email=None, password=None,
                               ekey=None, fb_access_token=None,
                               tw_oauth_token=None,
                               tw_oauth_token_secret=None, api_key=None):
        """user/get_session_token

        http://www.mediafire.com/developers/core_api/1.3/user/#get_session_token
        """

        if app_id is None:
            raise ValueError("app_id must be defined")

        params = QueryParams({
            'application_id': str(app_id),
            'token_version': 2,
            'response_format': 'json'
        })

        if fb_access_token:
            params['fb_access_token'] = fb_access_token
            signature_keys = ['fb_access_token']
        elif tw_oauth_token and tw_oauth_token_secret:
            params['tw_oauth_token'] = tw_oauth_token
            params['tw_oauth_token_secret'] = tw_oauth_token_secret
            signature_keys = ['tw_oauth_token',
                              'tw_oauth_token_secret']
        elif (email or ekey) and password:
            signature_keys = []
            if email:
                signature_keys.append('email')
                params['email'] = email

            if ekey:
                signature_keys.append('ekey')
                params['ekey'] = ekey

            params['password'] = password
            signature_keys.append('password')
        else:
            raise ValueError("Credentials not provided")

        signature_keys.append('application_id')

        signature = hashlib.sha1()
        for key in signature_keys:
            signature.update(str(params[key]).encode('ascii'))

        # Note: If the app uses a callback URL to provide its API key,
        # or if it does not have the "Require Secret Key" option checked,
        # then the API key may be omitted from the signature
        if api_key:
            signature.update(api_key.encode('ascii'))

        query = urlencode(params)
        query += '&signature=' + signature.hexdigest()

        return self.request('user/get_session_token', params=query)