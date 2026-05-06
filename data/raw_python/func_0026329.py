def _read_metadata(self):
        """ Read the main metadata packaged within a bpch file, indicating
        the output filetype and its title.

        """

        filetype = self.fp.readline().strip()
        filetitle = self.fp.readline().strip()
        # Decode to UTF string, if possible
        try:
            filetype = str(filetype, 'utf-8')
            filetitle = str(filetitle, 'utf-8')
        except:
            # TODO: Handle this edge-case of converting file metadata more elegantly.
            pass

        self.__setattr__('filetype', filetype)
        self.__setattr__('filetitle', filetitle)