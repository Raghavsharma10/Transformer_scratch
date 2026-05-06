def run(exe, args, capturestd=False, env=None):
        """
        Runs an executable with an array of arguments, optionally in the
        specified environment.
        Returns stdout and stderr
        """
        command = [exe] + args
        if env:
            log.info("Executing [custom environment]: %s", " ".join(command))
        else:
            log.info("Executing : %s", " ".join(command))
        start = time.time()

        # Temp files will be automatically deleted on close()
        # If run() throws the garbage collector should call close(), so don't
        # bother with try-finally
        outfile = None
        errfile = None
        if capturestd:
            outfile = tempfile.TemporaryFile()
            errfile = tempfile.TemporaryFile()

        # Use call instead of Popen so that stdin is connected to the console,
        # in case user input is required
        # On Windows shell=True is needed otherwise the modified environment
        # PATH variable is ignored. On Unix this breaks things.
        r = subprocess.call(
            command, env=env, stdout=outfile, stderr=errfile, shell=WINDOWS)

        stdout = None
        stderr = None
        if capturestd:
            outfile.seek(0)
            stdout = outfile.read()
            outfile.close()
            errfile.seek(0)
            stderr = errfile.read()
            errfile.close()

        end = time.time()
        if r != 0:
            log.error("Failed [%.3f s]", end - start)
            raise RunException(
                "Non-zero return code", exe, args, r, stdout, stderr)
        log.info("Completed [%.3f s]", end - start)
        return stdout, stderr