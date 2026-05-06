def setProperty(self, orgresource, protect, dummy = 7046):
        """SetProperty

        Args:
            orgresource: File path
            protect: 'Y' or 'N', 중요 표시

        Returns:
            Integer number: # of version list
            False: Failed to get property

        """

        url = nurls['setProperty']

        data = {'userid': self.user_id,
                'useridx': self.useridx,
                'orgresource': orgresource,
                'protect': protect,
                'dummy': dummy,
                }

        r = self.session.post(url = url, data = data)

        return resultManager(r.text)