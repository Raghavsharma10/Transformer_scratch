def auth(self):
        """
        Auth is used to call the AUTH API of CricketAPI.
       
        Access token required for every request call to CricketAPI.
        Auth functional will post user Cricket API app details to server
        and return the access token.

        Return:
            Access token    
        """
        if not self.store_handler.has_value('access_token'):
            params = {}
            params["access_key"] = self.access_key
            params["secret_key"] = self.secret_key
            params["app_id"] = self.app_id
            params["device_id"] = self.device_id
            auth_url = self.api_path + "auth/"
            response = self.get_response(auth_url, params, "post")

            if 'auth' in response:
                self.store_handler.set_value("access_token", response['auth']['access_token'])
                self.store_handler.set_value("expires", response['auth']['expires'])
                logger.info('Getting new access token')
            else:
                msg = "Error getting access_token, " + \
                      "please verify your access_key, secret_key and app_id"
                logger.error(msg)
                raise Exception("Auth Failed, please check your access details")