def load_model(self, name=None):
        '''
        Loads a saved version of the model.

        '''

        if self.clobber:
            return False

        if name is None:
            name = self.name
        file = os.path.join(self.dir, '%s.npz' % name)
        if os.path.exists(file):
            if not self.is_parent:
                log.info("Loading '%s.npz'..." % name)
            try:
                data = np.load(file)
                for key in data.keys():
                    try:
                        setattr(self, key, data[key][()])
                    except NotImplementedError:
                        pass

                # HACK: Backwards compatibility. Previous version stored
                # the CDPP in the `cdpp6`
                # and `cdpp6_arr` attributes. Let's move them over.
                if hasattr(self, 'cdpp6'):
                    self.cdpp = self.cdpp6
                    del self.cdpp6
                if hasattr(self, 'cdpp6_arr'):
                    self.cdpp_arr = np.array(self.cdpp6_arr)
                    del self.cdpp6_arr
                if hasattr(self, 'gppp'):
                    self.cdppg = self.gppp
                    del self.gppp

                # HACK: At one point we were saving the figure instances,
                # so loading the .npz
                # opened a plotting window. I don't think this is the case
                # any more, so this
                # next line should be removed in the future...
                pl.close()

                return True
            except:
                log.warn("Error loading '%s.npz'." % name)
                exctype, value, tb = sys.exc_info()
                for line in traceback.format_exception_only(exctype, value):
                    ln = line.replace('\n', '')
                    log.warn(ln)
                os.rename(file, file + '.bad')

        if self.is_parent:
            raise Exception(
                'Unable to load `%s` model for target %d.'
                % (self.name, self.ID))

        return False