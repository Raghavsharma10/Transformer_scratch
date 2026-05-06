def get_active_token(self):
        """
        Getting the valid access token.

           Access token expires every 24 hours, It will expires then it will
           generate a new token.
        Return:
           active access token 
        """

        expire_time = self.store_handler.has_value("expires")
        access_token = self.store_handler.has_value("access_token")
        if expire_time and access_token:
            expire_time = self.store_handler.get_value("expires")
            if not datetime.now() < datetime.fromtimestamp(float(expire_time)):
                self.store_handler.delete_value("access_token")
                self.store_handler.delete_value("expires")
                logger.info('Access token expired, going to get new token')
                self.auth()
            else:
                logger.info('Access token noy expired yet')
        else:
            self.auth()
        return self.store_handler.get_value("access_token")