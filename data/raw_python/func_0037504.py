def getProperty(self, full_path, dummy = 56184):
        """Get a file property

        :param full_path: The full path to get the file or directory property.

        :return: ``metadata`` if success or ``False`` if failed to get property

        :metadata:
              - creationdate
              - exif
              - filelink
              - filelinkurl
              - filetype => 1: document, 2: image, 3: video, 4: music, 5: zip
              - fileuploadstatus
              - getcontentlength
              - getlastmodified
              - href
              - lastaccessed
              - protect
              - resourceno
              - resourcetype
              - thumbnail
              - totalfilecnt
              - totalfoldercnt
              - virusstatus
        """
        data = {'orgresource': full_path,
                'userid': self.user_id,
                'useridx': self.useridx,
                'dummy': dummy,
                }

        s, metadata = self.POST('getProperty', data)

        if s is True:
            return metadata
        else:
            return False