def delete(self, file_path):
        """DELETE

        Args:
            file_path: Full path for a file you want to delete 
            upload_path: Ndrive path where you want to delete file
                ex) /Picture/

        Returns:
            True: Delete success
            False: Delete failed

        """
        now = datetime.datetime.now().isoformat()
        url = nurls['put'] + upload_path + file_name

        headers = {'userid': self.user_id,
                   'useridx': self.useridx,
                   'Content-Type': "application/x-www-form-urlencoded; charset=UTF-8",
                   'charset': 'UTF-8',
                   'Origin': 'http://ndrive2.naver.com',
        }
        r = self.session.delete(url = url, headers = headers)

        return self.resultManager(r.text)