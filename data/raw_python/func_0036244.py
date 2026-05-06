def _download(self, fmfile, destination, overwrite, callback):
        """The actual downloader streaming content from Filemail.

        :param fmfile: to download
        :param destination: destination path
        :param overwrite: replace existing files?
        :param callback: callback function that will receive total file size
         and written bytes as arguments
        :type fmfile: ``dict``
        :type destination: ``str`` or ``unicode``
        :type overwrite: ``bool``
        :type callback: ``func``
        """

        fullpath = os.path.join(destination, fmfile.get('filename'))
        path, filename = os.path.split(fullpath)

        if os.path.exists(fullpath):
            msg = 'Skipping existing file: {filename}'
            logger.info(msg.format(filename=filename))
            return

        filesize = fmfile.get('filesize')

        if not os.path.exists(path):
            os.makedirs(path)

        url = fmfile.get('downloadurl')
        stream = self.session.get(url, stream=True)

        def pg_callback(bytes_written):
            if pm.COMMANDLINE:
                bar.show(bytes_written)

            elif callback is not None:
                callback(filesize, bytes_written)

        if pm.COMMANDLINE:
            label = fmfile['filename'] + ': '
            bar = ProgressBar(label=label, expected_size=filesize)

        bytes_written = 0
        with open(fullpath, 'wb') as f:
            for chunk in stream.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    break

                f.write(chunk)
                bytes_written += len(chunk)

                # Callback
                pg_callback(bytes_written)