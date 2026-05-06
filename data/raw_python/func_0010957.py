def vanity(self):
        """ Returns the user's vanity url if it exists, None otherwise """
        purl = self.profile_url.strip('/')
        if purl.find("/id/") != -1:
            return os.path.basename(purl)