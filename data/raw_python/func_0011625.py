def get_access_token(self, code, client_id, client_secret):
        ''' 
        Exchange a temporary code for an access token allowing access to a user's account

        See https://developer.wunderlist.com/documentation/concepts/authorization for more info
        '''
        headers = {
                'Content-Type' : 'application/json'
                }
        data = {
                'client_id' : client_id,
                'client_secret' : client_secret,
                'code' : code,
                }
        str_data = json.dumps(data)
        response = requests.request(method='POST', url=ACCESS_TOKEN_URL, headers=headers, data=str_data)
        status_code = response.status_code
        if status_code != 200:
            raise ValueError("{} -- {}".format(status_code, response.json()))
        return body['access_token']