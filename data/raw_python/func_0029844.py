def record_to_fh(self, f):
        """Write the record, in filesystem format, to a file handle or file object"""

        fr = self.record

        if fr.contents:
            yaml.safe_dump(fr.unpacked_contents, f, default_flow_style=False, encoding='utf-8')
            fr.source_hash = self.fs_hash
            fr.modified = self.fs_modtime