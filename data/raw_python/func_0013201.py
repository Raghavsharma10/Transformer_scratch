def get_students(self, **kwargs):
        """Get users by promotion id.

        :param int promotion: Promotion ID
        :return: JSON
        """

        _promotion_id = kwargs.get('promotion')
        _url = PROMOTION_URL.format(promo_id=_promotion_id)
        return self._request_api(url=_url).json()