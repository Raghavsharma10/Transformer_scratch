def save_to(self, file):
        """Save data to file.

        Will copy by either writing out the data or using
        :func:`shutil.copyfileobj`.

        :param file: A file-like object (with a ``write`` method) or a
                     filename."""
        dest = file

        if hasattr(dest, 'write'):
            # writing to a file-like
            # only works when no unicode conversion is done
            if self.file is not None and\
                    getattr(self.file, 'encoding', None) is None:
                copyfileobj(self.file, dest)
            elif self.filename is not None:
                with open(self.filename, 'rb') as inp:
                    copyfileobj(inp, dest)
            else:
                dest.write(self.__bytes__())
        else:
            # we do not use filesystem io to make sure we have the same
            # permissions all around
            # copyfileobj() should be efficient enough

            # destination is a filename
            with open(dest, 'wb') as out:
                return self.save_to(out)