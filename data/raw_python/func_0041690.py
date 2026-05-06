def handle_two_factor_check(self, html: str) -> requests.Response:
        """
        Handling two factor authorization request
        """
        action_url = get_base_url(html)
        code = input(self.TWO_FACTOR_PROMPT).strip()
        data = {'code': code, '_ajax': '1', 'remember': '1'}
        post_url = '/'.join((self.LOGIN_URL, action_url))
        return self.post(post_url, data)