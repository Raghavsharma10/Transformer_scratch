def setProperty(self, full_path, protect, dummy = 7046):
        """Set property of a file.

        :param full_path: The full path to get the file or directory property.
        :param protect: 'Y' or 'N', 중요 표시

        :return: ``True`` when success to set property or ``False``
        """
        data = {'orgresource': full_path,
                'protect': protect,
                'userid': self.user_id,
                'useridx': self.useridx,
                'dummy': dummy,
                }
        s, metadata = self.POST('setProperty', data)

        if s is True:
            return True
        else:
            return False