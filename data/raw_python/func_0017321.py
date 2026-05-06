def _revert_tag_in_init(self):
        """
        Write version to __init__.py by editing in place
        """
        for line in fileinput.input(self.init_file, inplace=1):
            if line.strip().startswith("__version__"):
                line = "__version__ = \"" + self.init_version + "\""
            print(line.strip("\n"))
        
        print("reverted __init__.__version__ back to {}"
              .format(self.init_version))