def request(self, url):
        """
        Send a http request to the given *url*, try to decode
        the reply assuming it's JSON in UTF-8, and return the result

        :returns: Decoded result, or None in case of an error
        :rtype: mixed
        """
        self.logger.debug('url:\n' + url)
        try:
            response = urlopen(url)
            return json.loads(response.read().decode('utf-8'))
        except URLError:
            self.logger.info('Server connection problem')
        except Exception:
            self.logger.info('Server format problem')