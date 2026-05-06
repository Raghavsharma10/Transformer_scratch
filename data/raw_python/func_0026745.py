def get_entry_text(self, fname):
        """Retrieve the raw text from a file."""
        if fname.split('.')[-1] == 'gz':
            with gz.open(fname, 'rt') as f:
                filetext = f.read()
        else:
            with codecs.open(fname, 'r') as f:
                filetext = f.read()
        return filetext