def check_for_additional_actions(self, url_params: dict,
                                     html: str,
                                     login_data: dict) -> None:
        """
        Checks the url for a request for additional actions,
        if so, calls the event handler
        """
        action_response = ''
        if 'sid' in url_params:
            action_response = self.handle_captcha(url_params, html, login_data)
        elif 'authcheck' in url_params:
            action_response = self.handle_two_factor_check(html)
        elif 'security_check' in url_params:
            action_response = self.handle_phone_number_check(html)
        if action_response:
            check_page_for_warnings(action_response.text)