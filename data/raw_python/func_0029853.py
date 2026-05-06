def import_module(self, module_path = 'ambry.build', **kwargs):
        """
        Import the contents of the file into the ambry.build module

        :param kwargs: items to add to the module globals
        :return:
        """
        from fs.errors import NoSysPathError

        if module_path in sys.modules:
            module = sys.modules[module_path]
        else:
            module = imp.new_module(module_path)
            sys.modules[module_path] = module

        bf = self.record

        if not bf.contents:
            return module

        module.__dict__.update(**kwargs)

        try:
            abs_path = self._fs.getsyspath(self.file_name)
        except NoSysPathError:
            abs_path = '<string>'

        import re

        if re.search(r'-\*-\s+coding:', bf.contents):
            # Has encoding, so don't decode
            contents = bf.contents
        else:
            contents = bf.unpacked_contents  # Assumes utf-8

        exec(compile(contents, abs_path, 'exec'), module.__dict__)

        return module