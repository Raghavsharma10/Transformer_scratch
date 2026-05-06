def user_info(self):
        """
        General user info information as returned in /account in our API
        """
        account_uri = self.uri.split('/api/v1')[0] + '/account'
        req = self.request(account_uri)
        account_info = req.get()
        user_data = HTMLParser(account_info.content)
        return user_data