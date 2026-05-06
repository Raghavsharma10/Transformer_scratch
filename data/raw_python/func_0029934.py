def cast_to_subclass(self):
        """
        Load the bundle file from the database to get the derived bundle class,
        then return a new bundle built on that class

        :return:
        """
        self.import_lib()
        self.load_requirements()

        try:
            self.commit()  # To ensure the rollback() doesn't clear out anything important
            bsf = self.build_source_files.file(File.BSFILE.BUILD)
        except Exception as e:
            self.log('Error trying to create a bundle source file ... {} '.format(e))
            raise
            self.rollback()
            return self

        try:
            clz = bsf.import_bundle()

        except Exception as e:

            raise BundleError('Failed to load bundle code file, skipping : {}'.format(e))

        b = clz(self._dataset, self._library, self._source_url, self._build_url)
        b.limited_run = self.limited_run
        b.capture_exceptions = self.capture_exceptions
        b.multi = self.multi


        return b