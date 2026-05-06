def login(
            self):
        """login"""

        auth_url = self.api_urls["login"]

        if self.verbose:
            log.info(("log in user={} url={} ca_dir={} cert={}")
                     .format(
                        self.user,
                        auth_url,
                        self.ca_dir,
                        self.cert))

        use_headers = {
            "Content-type": "application/json"
        }
        login_data = {
            "username": self.user,
            "password": self.password
        }

        if self.debug:
            log.info((
                "LOGIN with body={} headers={} url={} "
                "verify={} cert={}").format(
                    login_data,
                    use_headers,
                    auth_url,
                    self.use_verify,
                    self.cert))

        response = requests.post(
            auth_url,
            verify=self.use_verify,
            cert=self.cert,
            data=json.dumps(login_data),
            headers=use_headers)

        if self.debug:
            log.info(("LOGIN response status_code={} text={} reason={}")
                     .format(
                        response.status_code,
                        response.text,
                        response.reason))

        user_token = ""
        if response.status_code == 200:
            user_token = json.loads(response.text)["token"]

        if user_token != "":
            self.token = user_token
            self.login_status = LOGIN_SUCCESS

            if self.verbose:
                log.debug("login success")
        else:
            log.error(("failed to login user={} to url={} text={}")
                      .format(
                        self.user,
                        auth_url,
                        response.text))
            self.login_status = LOGIN_FAILED
        # if the user token exists

        return self.login_status