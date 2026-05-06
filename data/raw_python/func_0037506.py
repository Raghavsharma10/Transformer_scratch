def getMusicAlbumList(self, tagtype = 0, startnum = 0, pagingrow = 100, dummy = 51467):
        """Get music album list.

        :param tagtype: ?

        :return: ``metadata`` or ``False``

        :metadata:
            - u'album':u'Greatest Hits Coldplay',
            - u'artist':u'Coldplay',
            - u'href':u'/Coldplay - Clocks.mp3',
            - u'musiccount':1,
            - u'resourceno':12459548378,
            - u'tagtype':1,
            - u'thumbnailpath':u'N',
            - u'totalpath':u'/'
        """
        data = {'tagtype': tagtype,
                'startnum': startnum,
                'pagingrow': pagingrow,
                'userid': self.user_id,
                'useridx': self.useridx,
                'dummy': dummy,
                }
        s, metadata = self.POST('getMusicAlbumList', data)

        if s is True:
            return metadata
        else:
            return False