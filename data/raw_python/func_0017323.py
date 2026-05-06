def build_conda_packages(self):
        """
        Run the Linux build and use converter to build OSX
        """
        ## check if update is necessary
        #if self.nversion == self.pversion:
        #    raise SystemExit("Exited: new version == existing version")

        ## tmp dir
        bldir = "./tmp-bld"
        if not os.path.exists(bldir):
            os.makedirs(bldir)

        ## iterate over builds
        for pybuild in ["2.7", "3"]:

            ## build and upload Linux to anaconda.org
            build = api.build(
                "conda-recipe/{}".format(self.package),
                 python=pybuild)
                
            ## upload Linux build
            if not self.deploy:
                cmd = ["anaconda", "upload", build[0], "--label", "test", "--force"]
            else:
                cmd = ["anaconda", "upload", build[0]]
            err = subprocess.Popen(cmd).communicate()

            ## build OSX copies 
            api.convert(build[0], output_dir=bldir, platforms=["osx-64"])
            osxdir = os.path.join(bldir, "osx-64", os.path.basename(build[0]))
            if not self.deploy:
                cmd = ["anaconda", "upload", osxdir, "--label", "test", "--force"]
            else:
                cmd = ["anaconda", "upload", osxdir]
            err = subprocess.Popen(cmd).communicate()

        ## cleanup tmpdir
        shutil.rmtree(bldir)