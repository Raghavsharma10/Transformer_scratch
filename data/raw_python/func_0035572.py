def ctox(self):
        """Main method for the environment.

        Parse the tox.ini config, install the dependancies and run the
        commands. The output of the commands is printed.

        Returns 0 if they ran successfully, 1 if there was an error
        (either in setup or whilst running the commands), 2 if the build
        was skipped.

        """
        # TODO make this less of a hack e.g. using basepython from config
        # if it exists (and use an attribute directly).
        if self.name[:4] not in SUPPORTED_ENVS:
            from colorama import Style
            cprint(Style.BRIGHT +
                   "Skipping unsupported python version %s\n" % self.name,
                   'warn')
            return 2

        # TODO don't remove env if there's a dependancy mis-match
        # rather "clean" it to the empty state (the hope being to keep
        # the dist build around - so not all files need to be rebuilt)
        # TODO extract this as a method (for readability)
        if not self.env_exists() or self.reusableable():
            cprint("%s create: %s" % (self.name, self.envdir))
            self.create_env(force_remove=True)

            cprint("%s installdeps: %s" % (self.name, ', '.join(self.deps)))
            if not self.install_deps():
                cprint("    deps installation failed, aborted.\n", 'err')
                return 1
        else:
            cprint("%s cached (deps unchanged): %s" % (self.name, self.envdir))

        # install the project from the zipped file
        # TODO think more carefully about where it should be installed
        # specifically we want to be able this to include the test files (which
        # are not always unpacked when installed so as to run the tests there)
        # if there are build files (e.g. cython) then tests must run where
        # the build was. Also, reinstalling should not overwrite the builds
        # e.g. setup.py will skip rebuilding cython files if they are unchanged
        cprint("%s inst: %s" % (self.name, self.envdistdir))
        if not self.install_dist():
            cprint("    install failed.\n", 'err')
            return 1

        cprint("%s runtests" % self.name)
        # return False if all commands were successfully run
        # otherwise returns True if at least one command exited badly
        return self.run_commands()