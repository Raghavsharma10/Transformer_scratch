def write(self, path, prog=None, format='raw'):
        """Write graph to file in selected format.

        Given a filename 'path' it will open/create and truncate
        such file and write on it a representation of the graph
        defined by the dot object and in the format specified by
        'format'. 'path' can also be an open file-like object, such as
        a StringIO instance.

        The format 'raw' is used to dump the string representation
        of the Dot object, without further processing.
        The output can be processed by any of graphviz tools, defined
        in 'prog', which defaults to 'dot'
        Returns True or False according to the success of the write
        operation.

        There's also the preferred possibility of using:

            write_'format'(path, prog='program')

        which are automatically defined for all the supported formats.
        [write_ps(), write_gif(), write_dia(), ...]

        """
        if prog is None:
            prog = self.prog

        fobj, close = get_fobj(path, 'w+b')
        try:
            if format == 'raw':
                data = self.to_string()
                if isinstance(data, basestring):
                    if not isinstance(data, unicode):
                        try:
                            data = unicode(data, 'utf-8')
                        except Exception:
                            pass

                try:
                    charset = self.get_charset()
                    if not PY3 or not charset:
                        charset = 'utf-8'
                    data = data.encode(charset)
                except Exception:
                    if PY3:
                        data = data.encode('utf-8')
                    pass

                fobj.write(data)

            else:
                fobj.write(self.create(prog, format))
        finally:
            if close:
                fobj.close()

        return True