def write_text(self, text, encoding=None, errors='strict',
                   linesep=os.linesep, append=False):
        r""" Write the given text to this file.

        The default behavior is to overwrite any existing file;
        to append instead, use the `append=True` keyword argument.

        There are two differences between :meth:`write_text` and
        :meth:`write_bytes`: newline handling and Unicode handling.
        See below.

        Parameters:

          `text` - str/unicode - The text to be written.

          `encoding` - str - The Unicode encoding that will be used.
              This is ignored if `text` isn't a Unicode string.

          `errors` - str - How to handle Unicode encoding errors.
              Default is ``'strict'``.  See ``help(unicode.encode)`` for the
              options.  This is ignored if `text` isn't a Unicode
              string.

          `linesep` - keyword argument - str/unicode - The sequence of
              characters to be used to mark end-of-line.  The default is
              :data:`os.linesep`.  You can also specify ``None`` to
              leave all newlines as they are in `text`.

          `append` - keyword argument - bool - Specifies what to do if
              the file already exists (``True``: append to the end of it;
              ``False``: overwrite it.)  The default is ``False``.


        --- Newline handling.

        ``write_text()`` converts all standard end-of-line sequences
        (``'\n'``, ``'\r'``, and ``'\r\n'``) to your platform's default
        end-of-line sequence (see :data:`os.linesep`; on Windows, for example,
        the end-of-line marker is ``'\r\n'``).

        If you don't like your platform's default, you can override it
        using the `linesep=` keyword argument.  If you specifically want
        ``write_text()`` to preserve the newlines as-is, use ``linesep=None``.

        This applies to Unicode text the same as to 8-bit text, except
        there are three additional standard Unicode end-of-line sequences:
        ``u'\x85'``, ``u'\r\x85'``, and ``u'\u2028'``.

        (This is slightly different from when you open a file for
        writing with ``fopen(filename, "w")`` in C or ``open(filename, 'w')``
        in Python.)


        --- Unicode

        If `text` isn't Unicode, then apart from newline handling, the
        bytes are written verbatim to the file.  The `encoding` and
        `errors` arguments are not used and must be omitted.

        If `text` is Unicode, it is first converted to :func:`bytes` using the
        specified `encoding` (or the default encoding if `encoding`
        isn't specified).  The `errors` argument applies only to this
        conversion.

        """
        if isinstance(text, text_type):
            if linesep is not None:
                text = U_NEWLINE.sub(linesep, text)
            text = text.encode(encoding or sys.getdefaultencoding(), errors)
        else:
            assert encoding is None
            text = NEWLINE.sub(linesep, text)
        self.write_bytes(text, append=append)