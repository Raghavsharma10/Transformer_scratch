def tox_get_python_executable(envconfig):
    """Return a python executable for the given python base name.

    The first plugin/hook which returns an executable path will determine it.

    ``envconfig`` is the testenv configuration which contains
    per-testenv configuration, notably the ``.envname`` and ``.basepython``
    setting.
    """
    try:
        # pylint: disable=no-member
        pyenv = (getattr(py.path.local.sysfind('pyenv'), 'strpath', 'pyenv')
                 or 'pyenv')
        cmd = [pyenv, 'which', envconfig.basepython]
        pipe = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        out, err = pipe.communicate()
    except OSError:
        err = '\'pyenv\': command not found'
        LOG.warning(
            "pyenv doesn't seem to be installed, you probably "
            "don't want this plugin installed either."
        )
    else:
        if pipe.poll() == 0:
            return out.strip()
        else:
            if not envconfig.tox_pyenv_fallback:
                raise PyenvWhichFailed(err)
    LOG.debug("`%s` failed thru tox-pyenv plugin, falling back. "
              "STDERR: \"%s\" | To disable this behavior, set "
              "tox_pyenv_fallback=False in your tox.ini or use "
              " --tox-pyenv-no-fallback on the command line.",
              ' '.join([str(x) for x in cmd]), err)