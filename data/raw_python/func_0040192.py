def extract_credentials(self, url):
        """
        Extracts user/password from a url.

        Returns a tuple:
            (url-without-auth, username, password)
        """
        if isinstance(url, urllib2.Request):
            result = urlparse.urlsplit(url.get_full_url())
        else:
            result = urlparse.urlsplit(url)
        scheme, netloc, path, query, frag = result

        username, password = self.parse_credentials(netloc)
        if username is None:
            return url, None, None
        elif password is None and self.prompting:
            # remove the auth credentials from the url part
            netloc = netloc.replace('%s@' % username, '', 1)
            # prompt for the password
            prompt = 'Password for %s@%s: ' % (username, netloc)
            password = urllib.quote(getpass.getpass(prompt))
        else:
            # remove the auth credentials from the url part
            netloc = netloc.replace('%s:%s@' % (username, password), '', 1)

        target_url = urlparse.urlunsplit((scheme, netloc, path, query, frag))
        return target_url, username, password