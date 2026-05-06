def makeDirectory(self, full_path, dummy = 40841):
        """Make a directory

            >>> nd.makeDirectory('/test')
        
        :param full_path: The full path to get the directory property. Should be end with '/'.

        :return: ``True`` when success to make a directory or ``False``
        """
        if full_path[-1] is not '/':
            full_path += '/'

        data = {'dstresource': full_path,
                'userid': self.user_id,
                'useridx': self.useridx,
                'dummy': dummy,
                }

        s, metadata = self.POST('makeDirectory', data)

        return s