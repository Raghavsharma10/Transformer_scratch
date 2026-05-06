def add_chassis(self, chassis):
        """
        :param chassis: chassis object
        """

        res = self._request(RestMethod.post, self.user_url, params={'ip': chassis.ip, 'port': chassis.port})
        assert(res.status_code == 201)