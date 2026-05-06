def temp_saved(self, suffix='', prefix='tmp', dir=None):
        """Saves data to temporary file and returns the relevant instance of
        :func:`~tempfile.NamedTemporaryFile`. The resulting file is not
        deleted upon closing, but when the context manager exits.

        Other arguments are passed on to :func:`~tempfile.NamedTemporaryFile`.
        """
        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=prefix,
            dir=dir,
            delete=False,
        )

        try:
            self.save_to(tmp)
            tmp.flush()
            tmp.seek(0)
            yield tmp
        finally:
            try:
                os.unlink(tmp.name)
            except OSError as e:
                if e.errno != 2:
                    reraise(e)