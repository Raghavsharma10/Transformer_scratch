def doMove(self, from_path, to_path, overwrite = False, bShareFireCopy = 'false', dummy = 56147):
        """Move a file.

            >>> nd.doMove('/Picture/flower.png', '/flower.png')

        :param from_path: The path to the file or folder to be moved.
        :param to_path: The destination path of the file or folder to be copied. File name should be included in the end of to_path.
        :param overwrite: Whether to overwrite an existing file at the given path. (Default ``False``.)
        :param bShareFireCopy: ???

        :return: ``True`` if success to move a file or ``False``.
        """
        if overwrite:
            overwrite = 'F'
        else:
            overwrite = 'T'

        data = {'orgresource': from_path,
                'dstresource': to_path,
                'overwrite': overwrite,
                'bShareFireCopy': bShareFireCopy,
                'userid': self.user_id,
                'useridx': self.useridx,
                'dummy': dummy,
                }

        s, metadata = self.POST('doMove', data)

        return s