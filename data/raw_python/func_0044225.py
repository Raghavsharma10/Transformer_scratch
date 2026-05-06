def compile_source(self, sourcepath):
        """
        Compile source to its destination

        Check if the source is eligible to compile (not partial and allowed
        from exclude patterns)

        Args:
            sourcepath (string): Sass source path to compile to its
                destination using project settings.

        Returns:
            tuple or None: A pair of (sourcepath, destination), if source has
                been compiled (or at least tried). If the source was not
                eligible to compile, return will be ``None``.
        """
        relpath = os.path.relpath(sourcepath, self.settings.SOURCES_PATH)

        conditions = {
            'sourcedir': None,
            'nopartial': True,
            'exclude_patterns': self.settings.EXCLUDES,
            'excluded_libdirs': self.settings.LIBRARY_PATHS,
        }
        if self.finder.match_conditions(sourcepath, **conditions):
            destination = self.finder.get_destination(
                relpath,
                targetdir=self.settings.TARGET_PATH
            )

            self.logger.debug(u"Compile: {}".format(sourcepath))
            success, message = self.compiler.safe_compile(
                self.settings,
                sourcepath,
                destination
            )

            if success:
                self.logger.info(u"Output: {}".format(message))
            else:
                self.logger.error(message)

            return sourcepath, destination

        return None