def load_module(self, fullname):
        """
        load_module is always called with the same argument as finder's
        find_module, see "How Import Works"
        """
        mod = super(JsonLoader, self).load_module(fullname)

        try:
            with codecs.open(self.cfg_file, 'r', 'utf-8') as f:
                mod.__dict__.update(json.load(f))
        except ValueError:
            # if raise here, traceback will contain ValueError
            self.e = "ValueError"
            self.err_msg = sys.exc_info()[1]

        if self.e == "ValueError":
            err_msg = "\nJson file not valid: "
            err_msg += self.cfg_file + '\n'
            err_msg += str(self.err_msg)
            raise InvalidJsonError(err_msg)

        return mod