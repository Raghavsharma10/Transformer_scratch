def tempfile(self, mode='wb', **args):
        "write the contents of the file to a tempfile and return the tempfile filename"
        tf = tempfile.NamedTemporaryFile(mode=mode)
        self.write(tf.name, mode=mode, **args)
        return tfn