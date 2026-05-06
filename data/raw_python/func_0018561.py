def setup_previous_omero_env(self, olddir, savevarsfile):
        """
        Create a copy of the current environment for interacting with the
        current OMERO server installation
        """
        env = self.get_environment(savevarsfile)

        def addpath(varname, p):
            if not os.path.exists(p):
                raise Exception("%s does not exist!" % p)
            current = env.get(varname)
            if current:
                env[varname] = p + os.pathsep + current
            else:
                env[varname] = p

        olddir = os.path.abspath(olddir)
        lib = os.path.join(olddir, "lib", "python")
        addpath("PYTHONPATH", lib)
        bin = os.path.join(olddir, "bin")
        addpath("PATH", bin)
        self.old_env = env