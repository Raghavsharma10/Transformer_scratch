def invoke(self, headers, body):
        """
        Invokes the soap service
        """
        xml = Service._create_request(headers, body)

        try:
            response = self.session.post(self.endpoint, verify=False, data=xml)
            logging.debug(response.content)
        except Exception as e:
            traceback.print_exc()
            raise WSManException(e)

        if response.status_code == 200:
            return Service._parse_response(response.content)

        if response.status_code == 401:
            raise WSManAuthenticationException('the remote host rejected authentication')

        raise WSManException('the remote host returned an unexpected http status code: %s' % response.status_code)