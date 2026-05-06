def visualize(self):
        """Launch kcachegrind on the converted entries.

        One of the executables listed in KCACHEGRIND_EXECUTABLES
        must be present in the system path.
        """

        available_cmd = None
        for cmd in KCACHEGRIND_EXECUTABLES:
            if is_installed(cmd):
                available_cmd = cmd
                break

        if available_cmd is None:
            sys.stderr.write("Could not find kcachegrind. Tried: %s\n" %
                             ", ".join(KCACHEGRIND_EXECUTABLES))
            return

        if self.out_file is None:
            fd, outfile = tempfile.mkstemp(".log", "pyprof2calltree")
            use_temp_file = True
        else:
            outfile = self.out_file.name
            use_temp_file = False

        try:
            if use_temp_file:
                with io.open(fd, "w") as f:
                    self.output(f)
            subprocess.call([available_cmd, outfile])
        finally:
            # clean the temporary file
            if use_temp_file:
                os.remove(outfile)
                self.out_file = None