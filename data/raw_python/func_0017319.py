def _write_new_tag_to_init(self):
        """
        Write version to __init__.py by editing in place
        """
        for line in fileinput.input(self.init_file, inplace=1):
            if line.strip().startswith("__version__"):
                line = "__version__ = \"" + self.tag + "\""
            print(line.strip("\n"))