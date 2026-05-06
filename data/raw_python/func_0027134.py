def __make_request(self, requests_session, method, url, params, data, headers, certificate):
        """ Encapsulate requests call
        """
        verify = False
        timeout = self.timeout

        try:  # TODO : Remove this ugly try/except after fixing Java issue: http://mvjira.mv.usa.alcatel.com/browse/VSD-546
            response = requests_session.request(method=method,
                                        url=url,
                                        data=data,
                                        headers=headers,
                                        verify=verify,
                                        timeout=timeout,
                                        params=params,
                                        cert=certificate)
        except requests.exceptions.SSLError:
            try:
                response = requests_session.request(method=method,
                                            url=url,
                                            data=data,
                                            headers=headers,
                                            verify=verify,
                                            timeout=timeout,
                                            params=params,
                                            cert=certificate)
            except requests.exceptions.Timeout:
                return self._did_timeout()

        except requests.exceptions.Timeout:
            return self._did_timeout()

        return response