def attempt_dev_link_via_import(self, egg):
        """Create egg-link to FS location if an egg is found through importing.

        Sometimes an egg *is* installed, but without a proper egg-info file.
        So we attempt to import the egg in order to return a link anyway.

        TODO: currently it only works with simple package names like
        "psycopg2" and "mapnik".

        """
        try:
            imported = __import__(egg)
        except ImportError:
            self.logger.warn("Tried importing '%s', but that also didn't work.", egg)
            self.logger.debug("For reference, sys.path is %s", sys.path)
            return
        self.logger.info("Importing %s works, however", egg)
        try:
            probable_location = os.path.dirname(imported.__file__)
        except:  # Bare except
            self.logger.exception("Determining the location failed, however")
            return
        filesystem_egg_link = os.path.join(
            self.dev_egg_dir,
            '%s.egg-link' % egg)
        f = open(filesystem_egg_link, 'w')
        f.write(probable_location)
        f.close()
        self.logger.info('Using sysegg %s for %s', probable_location, egg)
        self.added.append(filesystem_egg_link)
        return True