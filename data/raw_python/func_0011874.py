def ste(command, nindent, mdir, fpointer, env=None):
    """
    Print STDOUT of a shell command formatted in reStructuredText.

    This is a simplified version of :py:func:`pmisc.term_echo`.

    :param command: Shell command (relative to **mdir** if **env** is not given)
    :type  command: string

    :param nindent: Indentation level
    :type  nindent: integer

    :param mdir: Module directory, used if **env** is not given
    :type  mdir: string

    :param fpointer: Output function pointer. Normally is :code:`cog.out` but
                     :code:`print` or other functions can be used for
                     debugging
    :type  fpointer: function object

    :param env: Environment dictionary. If not provided, the environment
                dictionary is the key "PKG_BIN_DIR" with the value of the
                **mdir**
    :type  env: dictionary

    For example::

        .. This is a reStructuredText file snippet
        .. [[[cog
        .. import os, sys
        .. from docs.support.term_echo import term_echo
        .. file_name = sys.modules['docs.support.term_echo'].__file__
        .. mdir = os.path.realpath(
        ..     os.path.dirname(
        ..         os.path.dirname(os.path.dirname(file_name))
        ..     )
        .. )
        .. [[[cog ste('build_docs.py -h', 0, mdir, cog.out) ]]]

        .. code-block:: console

        $ ${PKG_BIN_DIR}/build_docs.py -h
        usage: build_docs.py [-h] [-d DIRECTORY] [-n NUM_CPUS]
        ...
        $

        .. ]]]

    """
    sdir = LDELIM + "PKG_BIN_DIR" + RDELIM
    command = (
        sdir + ("{sep}{cmd}".format(sep=os.path.sep, cmd=command))
        if env is None
        else command
    )
    env = {"PKG_BIN_DIR": mdir} if env is None else env
    term_echo(command, nindent, env, fpointer)