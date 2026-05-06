def _compile_files(self):
        """
        Compiles python plugin files in order to be processed by the loader.
        It compiles the plugins if they have been updated or haven't yet been
        compiled.
        """
        for f in glob.glob(os.path.join(self.dir_path, '*.py')):
            # Check for compiled Python files that aren't in the __pycache__.
            if not os.path.isfile(os.path.join(self.dir_path, f + 'c')):
                compileall.compile_dir(self.dir_path, quiet=True)
                logging.debug('Compiled plugins as a new plugin has been added.')
                return
            # Recompile if there are newer plugins.
            elif os.path.getmtime(os.path.join(self.dir_path, f)) > os.path.getmtime(
                    os.path.join(self.dir_path, f + 'c')):
                compileall.compile_dir(self.dir_path, quiet=True)
                logging.debug('Compiled plugins as a plugin has been changed.')
                return