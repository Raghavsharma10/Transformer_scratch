def write(self,
        fout=None,
        fmt=SPARSE,
        schema_only=False,
        data_only=False):
        """
        Write an arff structure to a string.
        """
        assert not (schema_only and data_only), 'Make up your mind.'
        assert fmt in FORMATS, 'Invalid format "%s". Should be one of: %s' % (fmt, ', '.join(FORMATS))
        close = False
        if fout is None:
            close = True
            fout = StringIO()
        if not data_only:
            print('% ' + re.sub("\n", "\n% ", '\n'.join(self.comment)), file=fout)
            print("@relation " + self.relation, file=fout)
            self.write_attributes(fout=fout)
        if not schema_only:
            print("@data", file=fout)
            for d in self.data:
                line_str = self.write_line(d, fmt=fmt)
                if line_str:
                    print(line_str, file=fout)
        if isinstance(fout, StringIO) and close:
            return fout.getvalue()