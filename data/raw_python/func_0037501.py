def downloadFile(self, from_path, to_path = ''):
        """Download a file.

            >>> nd.downloadFile('/Picture/flower.png', '~/flower.png')

        :param from_path: The full path to download the file to, *including the file name*. If the destination directory does not yet exist, it will be created.
        :param to_path: The full path of a file to be saved in local directory.

        :returns: File object
        """

        if to_path == '':
            file_name = os.path.basename(from_path)
            to_path = os.path.join(os.getcwd(), file_name)

        url = nurls['download'] + from_path

        data = {'attachment':2,
                'userid': self.user_id,
                'useridx': self.useridx,
                'NDriveSvcType': "NHN/ND-WEB Ver",
               }

        if '~' in to_path:
            to_path = expanduser(to_path)

        with open(to_path, 'wb') as handle:
            request = self.session.get(url, params = data, stream=True)

            for block in request.iter_content(1024):
                if not block:
                    break
                handle.write(block)
            return handle