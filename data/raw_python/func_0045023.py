def get_api_user_key(self, api_dev_key, username=None, password=None):
        '''
        Get api user key to enable posts from user accounts if username
        and password available.
        Not getting an api_user_key means that the posts will be "guest" posts
        '''
        username = username or get_config('pastebin', 'api_user_name')
        password = password or get_config('pastebin', 'api_user_password')
        if username and password:
            data = {
                'api_user_name': username,
                'api_user_password': password,
                'api_dev_key': api_dev_key,
            }
            urlencoded_data = urllib.urlencode(data)
            req = urllib2.Request('http://pastebin.com/api/api_login.php',
                                  urlencoded_data)
            response = urllib2.urlopen(req)
            user_key = response.read()
            logging.debug("User key: %s" % user_key)
            return user_key
        else:
            logging.info("Pastebin: not using any user key")
            return ""