def get_grades(self, login=None, promotion=None, **kwargs):
        """Get a user's grades on a single promotion based on his login.

        Either use the `login` param, or the client's login if unset.
        :return: JSON
        """

        _login = kwargs.get(
            'login',
            login or self._login
        )
        _promotion_id = kwargs.get('promotion', promotion)
        _grades_url = GRADES_URL.format(login=_login, promo_id=_promotion_id)
        return self._request_api(url=_grades_url).json()