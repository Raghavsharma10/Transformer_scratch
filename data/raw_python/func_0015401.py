def run(self):
        """
        Need to find any pre-existing vext contained in dependent packages
        and install them

        example:

        you create a setup.py with install_requires["vext.gi"]:

        - vext.gi gets installed using bdist_egg
        - vext itself is now called with bdist_egg and we end up here

        Vext now needs to find and install .vext files in vext.gi
        [or any other files that depend on vext]

        :return:
        """
        logger.debug("vext InstallLib [started]")

        # Find packages that depend on vext and check for .vext files...

        logger.debug("find_vext_files")
        vext_files = self.find_vext_files()
        logger.debug("manually_install_vext: ", vext_files)
        self.manually_install_vext(vext_files)
        logger.debug("enable vext")
        self.enable_vext()
        logger.debug("install_lib.run")
        install_lib.run(self)
        logger.debug("vext InstallLib [finished]")