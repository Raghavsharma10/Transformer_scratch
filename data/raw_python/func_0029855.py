def import_lib(self):
        """Import the lib.py file into the bundle module"""

        try:
            import ambry.build
            module = sys.modules['ambry.build']
        except ImportError:
            module = imp.new_module('ambry.build')
            sys.modules['ambry.build'] = module

        bf = self.record

        if not bf.has_contents:
            return

        try:
            exec (compile(bf.contents, self.path, 'exec'), module.__dict__)

        except Exception:
            self._bundle.error("Failed to load code from {}".format(self.path))
            raise

        # print(self.file_const, bundle.__dict__.keys())
        # print(bf.contents)

        return module