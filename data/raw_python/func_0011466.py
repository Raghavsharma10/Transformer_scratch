def define_natives(cls):
        """Define the native functions for PFP
        """
        if len(cls._natives) > 0:
            return

        glob_pattern = os.path.join(os.path.dirname(__file__), "native", "*.py")
        for filename in glob.glob(glob_pattern):
            basename = os.path.basename(filename).replace(".py", "")
            if basename == "__init__":
                continue

            try:
                mod_base = __import__("pfp.native", globals(), locals(), fromlist=[basename])
            except Exception as e:
                sys.stderr.write("cannot import native module {} at '{}'".format(basename, filename))
                raise e
                continue

            mod = getattr(mod_base, basename)
            setattr(mod, "PYVAL", fields.get_value)
            setattr(mod, "PYSTR", fields.get_str)