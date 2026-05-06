def checkStatus(self):
        """Check status

        Args:

        Returns:
            True: Sucess
            False: Failed

        """
        checkAccount()

        data = {'userid': self.user_id,
                'useridx': self.useridx
               }
        r = self.session.post(nurls['checkStatus'], data = data)

        p = re.compile(r'\<message\>(?P<message>.+)\</message\>')
        message = p.search(r.text).group('message')

        if message == 'success':
            return True
        else:
            return False