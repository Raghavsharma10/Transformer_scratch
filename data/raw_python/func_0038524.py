def read_file(cls, path):
        """Reads file at path and returns dict with separated frontmatter.
        See read() for more info on dict return value.
        """
        with open(path, encoding="utf-8") as file:
            file_contents = file.read()
            return cls.read(file_contents)