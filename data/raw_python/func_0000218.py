def setup_versioneer():
    """
    Generate (temporarily) versioneer.py file in project root directory
    :return:
    """
    try:
        # assume versioneer.py was generated using "versioneer install" command
        import versioneer
        versioneer.get_version()
    except ImportError:
        # it looks versioneer.py is missing
        # lets assume that versioneer package is installed
        # and versioneer binary is present in $PATH
        import subprocess
        try:
            # call versioneer install to generate versioneer.py
            subprocess.check_output(["versioneer", "install"])
        except OSError:
            # it looks versioneer is missing from $PATH
            # probably versioneer is installed in some user directory

            # query pip for list of files in versioneer package
            # line below is equivalen to putting result of
            #  "pip show -f versioneer" command to string output
            output = pip_command_output(["show", "-f", "versioneer"])

            # now we parse the results
            import os
            # find absolute path where *versioneer package* was installed
            # and store it in main_path
            main_path = [x[len("Location: "):] for x in output.splitlines()
                         if x.startswith("Location")][0]
            # find path relative to main_path where
            # *versioneer binary* was installed
            bin_path = [x[len("  "):] for x in output.splitlines()
                        if x.endswith(os.path.sep + "versioneer")][0]

            # exe_path is absolute path to *versioneer binary*
            exe_path = os.path.join(main_path, bin_path)
            # call versioneer install to generate versioneer.py
            # line below is equivalent to running in terminal
            # "python versioneer install"
            subprocess.check_output(["python", exe_path, "install"])