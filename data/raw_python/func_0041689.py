def handle_captcha(self, query_params: dict,
                       html: str,
                       login_data: dict) -> requests.Response:
        """
        Handling CAPTCHA request
        """
        check_url = get_base_url(html)
        captcha_url = '{}?s={}&sid={}'.format(self.CAPTCHA_URI,
                                              query_params['s'],
                                              query_params['sid'])
        login_data['captcha_sid'] = query_params['sid']
        login_data['captcha_key'] = input(self.CAPTCHA_INPUT_PROMPT
                                          .format(captcha_url))
        return self.post(check_url, login_data)