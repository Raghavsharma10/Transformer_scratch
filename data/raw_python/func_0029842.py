def fh_to_record(self, f):
        """Load a file in the filesystem into the file record"""
        import unicodecsv as csv

        fn_path = self.file_name

        fr = self.record
        fr.path = fn_path
        rows = []

        # NOTE. There were two cases here, for PY2 and PY3. Py two had
        # encoding='utf-8' in the reader. I've combined them b/c that's the default for
        # unicode csv, so it shouldn't be necessary.

        # Should probably be something like this:
        #if sys.version_info[0] >= 3:  # Python 3
        #    import csv
        #    f = open(self._fstor.syspath, 'rtU', encoding=encoding)
        #    reader = csv.reader(f)
        #else:  # Python 2
        #    import unicodecsv as csv
        #    f = open(self._fstor.syspath, 'rbU')
        #    reader = csv.reader(f, encoding=encoding)

        for row in csv.reader(f):
            row = [e if e.strip() != '' else None for e in row]
            if any(bool(e) for e in row):
                rows.append(row)
        try:
            fr.update_contents(msgpack.packb(rows), 'application/msgpack')
        except AssertionError:
            raise

        fr.source_hash = self.fs_hash
        fr.synced_fs = self.fs_modtime
        fr.modified = self.fs_modtime