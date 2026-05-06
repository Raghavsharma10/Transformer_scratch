def write(self, fn=None):
        """copy the zip file from its filename to the given filename."""
        fn = fn or self.fn
        if not os.path.exists(os.path.dirname(fn)):
            os.makedirs(os.path.dirname(fn))
        f = open(self.fn, 'rb')
        b = f.read()
        f.close()
        f = open(fn, 'wb')
        f.write(b)
        f.close()