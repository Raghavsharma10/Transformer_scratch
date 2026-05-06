def getDiskSpace(self, file_path, upload_path = '', overwrite = False):
        """getDiskSpace

        Args:
            file_path: Full path for a file you want to checkUpload
            upload_path: Ndrive path where you want to upload file
                ex) /Picture/

        Returns:
            True: Possible to upload a file with a given file_size
            False: Impossible to upload a file with a given file_size

        """

        self.checkAccount()

        url = nurls['checkUpload']

        file_size = os.stat(file_path).st_size
        file_name = os.path.basename(file_path)

        now = datetime.datetime.now().isoformat()

        data = {'userid': self.user_id,
                'useridx': self.useridx,
                'getlastmodified': now,
                'dstresource': upload_path + file_name,
                'overwrite': overwrite,
                'uploadsize': file_size,
               }
        r = self.session.post(nurls['getDiskSpace'], data = data)

        return resultManager(r.text)