def put(self, file_path, upload_path = ''):
        """PUT

        Args:
            file_path: Full path for a file you want to upload
            upload_path: Ndrive path where you want to upload file
                ex) /Picture/

        Returns:
            True: Upload success
            False: Upload failed

        """
        f = open(file_path, "r")
        c = f.read()

        file_name = os.path.basename(file_path)

        now = datetime.datetime.now().isoformat()
        url = nurls['put'] + upload_path + file_name

        headers = {'userid': self.user_id,
                   'useridx': self.useridx,
                   'MODIFYDATE': now,
                   'Content-Type': magic.from_file(file_path, mime=True),
                   'charset': 'UTF-8',
                   'Origin': 'http://ndrive2.naver.com',
        }
        r = self.session.put(url = url, data = c, headers = headers)

        return self.resultManager(r.text)