def should_raptorize(self, req, resp):
        """ Determine if this request should be raptorized.  Boolean. """

        if resp.status != "200 OK":
            return False

        content_type = resp.headers.get('Content-Type', 'text/plain').lower()
        if not 'html' in content_type:
            return False

        if random.random() > self.random_chance:
            return False

        if self.only_on_april_1st:
            now = datetime.datetime.now()
            if now.month != 20 and now.day != 1:
                return False

        return True