def set_defaults(self):
        """Add default content to any file record that is empty"""

        for const_name, c in file_classes.items():
            if c.multiplicity == '1':
                f = self.file(const_name)
                if not f.record.unpacked_contents:
                    f.setcontent(f.default)