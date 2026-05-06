def _generate_badge(self, subject, status):
        """
        Generate SVG for one badge via shields.io.

        :param subject: subject; left-hand side of badge
        :type subject: str
        :param status: status; right-hand side of badge
        :type status: str
        :return: badge SVG
        :rtype: str
        """
        url = 'https://img.shields.io/badge/%s-%s-brightgreen.svg' \
              '?style=flat&maxAge=3600' % (subject, status)
        logger.debug("Getting badge for %s => %s (%s)", subject, status, url)
        res = requests.get(url)
        if res.status_code != 200:
            raise Exception("Error: got status %s for shields.io badge: %s",
                            res.status_code, res.text)
        logger.debug('Got %d character response from shields.io', len(res.text))
        return res.text