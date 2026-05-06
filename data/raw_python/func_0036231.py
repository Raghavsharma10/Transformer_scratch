def add_files(self, files):
        """Add files and/or folders to transfer.
        If :class:`Transfer.compress` attribute is set to ``True``, files
        will get packed into a zip file before sending.

        :param files: Files or folders to send
        :type files: str, list
        """

        if isinstance(files, basestring):
            files = [files]

        zip_file = None
        if self.zip_:
            zip_filename = self._get_zip_filename()
            zip_file = ZipFile(zip_filename, 'w')

        for filename in files:
            if os.path.isdir(filename):
                for dirname, subdirs, filelist in os.walk(filename):
                    if dirname:
                        if self.zip_:
                            zip_file.write(dirname)

                    for fname in filelist:
                        filepath = os.path.join(dirname, fname)
                        if self.zip_:
                            zip_file.write(filepath)

                        else:
                            fmfile = self.get_file_specs(filepath,
                                                         keep_folders=True)
                            if fmfile['totalsize'] > 0:
                                self._files.append(fmfile)

            else:
                if self.zip_:
                    zip_file.write(filename)

                else:
                    fmfile = self.get_file_specs(filename)
                    self._files.append(fmfile)

        if self.zip_:
            zip_file.close()
            filename = zip_filename
            fmfile = self.get_file_specs(filename)
            self._files.append(fmfile)