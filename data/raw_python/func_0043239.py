def run(*args):
    """Load given `envfile` and run `command` with `params`"""

    if not args:
        args = sys.argv[1:]

    if len(args) < 2:
        print('Usage: runenv <envfile> <command> <params>')
        sys.exit(0)
    os.environ.update(create_env(args[0]))
    os.environ['_RUNENV_WRAPPED'] = '1'
    runnable_path = args[1]

    if not runnable_path.startswith(('/', '.')):
        runnable_path = spawn.find_executable(runnable_path)

    try:
        if not(stat.S_IXUSR & os.stat(runnable_path)[stat.ST_MODE]):
            print('File `%s is not executable' % runnable_path)
            sys.exit(1)
        return subprocess.check_call(
            args[1:], env=os.environ
        )
    except subprocess.CalledProcessError as e:
        return e.returncode