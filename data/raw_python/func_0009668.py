def update(self, nickname=None, **kwargs):
        """Update your own membership.

        Note that this fails on former groups.

        :param str nickname: new nickname
        :return: updated membership
        :rtype: :class:`~groupy.api.memberships.Member`
        """
        url = self.url + 'hips/update'
        payload = {
            'membership': {
                'nickname': nickname,
            },
        }
        payload['membership'].update(kwargs)
        response = self.session.post(url, json=payload)
        return Member(self, self.group_id, **response.data)