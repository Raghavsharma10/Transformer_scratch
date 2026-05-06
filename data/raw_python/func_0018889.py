def load_file(self, filename):
        """Read and return the content of the given file.

        If the current directory is not defined explicitly, the directory
        name is constructed with the actual simulation start date.  If
        such an directory does not exist, it is created immediately.
        """
        _defaultdir = self.DEFAULTDIR
        try:
            if not filename.endswith('.py'):
                filename += '.py'
            try:
                self.DEFAULTDIR = (
                    'init_' + hydpy.pub.timegrids.sim.firstdate.to_string('os'))
            except KeyError:
                pass
            filepath = os.path.join(self.currentpath, filename)
            with open(filepath) as file_:
                return file_.read()
        except BaseException:
            objecttools.augment_excmessage(
                'While trying to read the conditions file `%s`'
                % filename)
        finally:
            self.DEFAULTDIR = _defaultdir