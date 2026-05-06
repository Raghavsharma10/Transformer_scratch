def makeBenchmarkRunner(path, args):
    """
    Make a function that will run two Python processes serially: first one
    which calls the setup function from the given file, then one which calls
    the execute function from the given file.
    """
    def runner():
        return BenchmarkProcess.spawn(
            executable=sys.executable,
            args=['-Wignore'] + args,
            path=path.path,
            env=os.environ)
    return runner