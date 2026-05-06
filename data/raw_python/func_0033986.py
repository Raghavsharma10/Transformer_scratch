def load_dir(self, path):
        """
        import contents of a directory
        """

        def visit_path(arg, dirname, names):
            for name in names:
                fpath = os.path.join(dirname, name)
                new_path = fpath[len(path):]
                if os.path.isfile(fpath):
                    content = open(fpath, "rb").read()
                    self.touch(new_path, content)
                else:
                    self.mkdir(new_path)

        os.path.walk(path, visit_path, None)