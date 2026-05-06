def get_response(self, url, username=None, password=None):
        """
        does the dirty work of actually getting the rsponse object using urllib2
        and its HTTP auth builtins.
        """
        scheme, netloc, path, query, frag = urlparse.urlsplit(url)
        req = self.get_request(url)

        stored_username, stored_password = self.passman.find_user_password(None, netloc)
        # see if we have a password stored
        if stored_username is None:
            if username is None and self.prompting:
                username = urllib.quote(raw_input('User for %s: ' % netloc))
                password = urllib.quote(getpass.getpass('Password: '))
            if username and password:
                self.passman.add_password(None, netloc, username, password)
            stored_username, stored_password = self.passman.find_user_password(None, netloc)
        authhandler = urllib2.HTTPBasicAuthHandler(self.passman)
        opener = urllib2.build_opener(authhandler)
        # FIXME: should catch a 401 and offer to let the user reenter credentials
        return opener.open(req)