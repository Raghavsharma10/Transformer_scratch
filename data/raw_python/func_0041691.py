def handle_phone_number_check(self, html: str) -> requests.Response:
        """
        Handling phone number request
        """
        action_url = get_base_url(html)
        phone_number = input(self.PHONE_PROMPT)
        url_params = get_url_params(action_url)
        data = {'code': phone_number,
                'act': 'security_check',
                'hash': url_params['hash']}
        post_url = '/'.join((self.LOGIN_URL, action_url))
        return self.post(post_url, data)