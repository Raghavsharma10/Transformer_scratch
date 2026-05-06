def getMusicAlbumList(self, tagtype = 0, startnum = 0, pagingrow = 100):
        """GetMusicAlbumList

        Args:
            tagtype = ???
            startnum
            pagingrow

        Returns:
            ???
            False: Failed to get property

        """

        url = nurls['setProperty']

        data = {'userid': self.user_id,
                'useridx': self.useridx,
                'tagtype': tagtype,
                'startnum': startnum,
                'pagingrow': pagingrow,
                }

        r = self.session.post(url = url, data = data)

        return resultManager(r.text)