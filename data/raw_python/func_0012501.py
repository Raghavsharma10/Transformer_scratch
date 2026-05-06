def run(command, input=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=None, copy_local_env=False, **kwargs):
    """
    Cross platform compatible subprocess with CompletedProcess return.

    No formatting or encoding is performed on the output of subprocess, so it's
    output will appear the same on each version / interpreter as before.

    .. code:: python

        reusables.run('echo "hello world!', shell=True)
        # CPython 3.6
        # CompletedProcess(args='echo "hello world!', returncode=0,
        #                  stdout=b'"hello world!\\r\\n', stderr=b'')
        #
        # PyPy 5.4 (Python 2.7.10)
        # CompletedProcess(args='echo "hello world!', returncode=0L,
        # stdout='"hello world!\\r\\n')

    Timeout is only usable in Python 3.X, as it was not implemented before then,
    a NotImplementedError will be raised if specified on 2.x version of Python.

    :param command: command to run, str if shell=True otherwise must be list
    :param input: send something `communicate`
    :param stdout: PIPE or None
    :param stderr: PIPE or None
    :param timeout: max time to wait for command to complete
    :param copy_local_env: Use all current ENV vars in the subprocess as well
    :param kwargs: additional arguments to pass to Popen
    :return: CompletedProcess class
    """
    if copy_local_env:
        # Copy local env first and overwrite with anything manually specified
        env = os.environ.copy()
        env.update(kwargs.get('env', {}))
    else:
        env = kwargs.get('env')

    if sys.version_info >= (3, 5):
        return subprocess.run(command, input=input, stdout=stdout,
                              stderr=stderr, timeout=timeout, env=env,
                              **kwargs)

    # Created here instead of root level as it should never need to be
    # manually created or referenced
    class CompletedProcess(object):
        """A backwards compatible near clone of subprocess.CompletedProcess"""

        def __init__(self, args, returncode, stdout=None, stderr=None):
            self.args = args
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

        def __repr__(self):
            args = ['args={0!r}'.format(self.args),
                    'returncode={0!r}'.format(self.returncode),
                    'stdout={0!r}'.format(self.stdout) if self.stdout else '',
                    'stderr={0!r}'.format(self.stderr) if self.stderr else '']
            return "{0}({1})".format(type(self).__name__,
                                     ', '.join(filter(None, args)))

        def check_returncode(self):
            if self.returncode:
                if python_version < (2, 7):
                    raise subprocess.CalledProcessError(self.returncode,
                                                        self.args)
                raise subprocess.CalledProcessError(self.returncode,
                                                    self.args,
                                                    self.stdout)

    proc = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                            env=env, **kwargs)
    if PY3:
        out, err = proc.communicate(input=input, timeout=timeout)
    else:
        if timeout:
            raise NotImplementedError("Timeout is only available on Python 3")
        out, err = proc.communicate(input=input)
    return CompletedProcess(command, proc.returncode, out, err)