def term_echo(command, nindent=0, env=None, fpointer=None, cols=60):
    """
    Print STDOUT of a shell command formatted in reStructuredText.

    .. role:: bash(code)
        :language: bash

    :param command: Shell command
    :type  command: string

    :param nindent: Indentation level
    :type  nindent: integer

    :param env: Environment variable replacement dictionary. The
                command is pre-processed and any environment variable
                represented in the full notation (:bash:`${...}` in Linux and
                OS X or :bash:`%...%` in Windows) is replaced. The dictionary
                key is the environment variable name and the dictionary value
                is the replacement value. For example, if **command** is
                :code:`'${PYTHON_CMD} -m "x=5"'` and **env** is
                :code:`{'PYTHON_CMD':'python3'}` the actual command issued
                is :code:`'python3 -m "x=5"'`
    :type  env: dictionary

    :param fpointer: Output function pointer. Normally is :code:`cog.out` but
                     :code:`print` or other functions can be used for
                     debugging
    :type  fpointer: function object

    :param cols: Number of columns of output
    :type  cols: integer
    """
    # pylint: disable=R0204
    # Set argparse width so that output does not need horizontal scroll
    # bar in narrow windows or displays
    os.environ["COLUMNS"] = str(cols)
    command_int = command
    if env:
        for var, repl in env.items():
            command_int = command_int.replace('"' + LDELIM + var + RDELIM + '"', repl)
            command_int = command_int.replace(LDELIM + var + RDELIM, repl)
    tokens = command_int.split(" ")
    # Add Python interpreter executable for Python scripts on Windows since
    # the shebang does not work
    if (platform.system().lower() == "windows") and (
        tokens[0].endswith(".py")
    ):  # pragma: no cover
        tokens = [sys.executable] + tokens
    proc = subprocess.Popen(tokens, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = proc.communicate()[0]
    if sys.hexversion >= 0x03000000:  # pragma: no cover
        stdout = stdout.decode("utf-8")
    stdout = stdout.split("\n")
    indent = nindent * " "
    fpointer(os.linesep)
    fpointer("{0}.. code-block:: console{1}".format(indent, os.linesep))
    fpointer(os.linesep)
    fpointer("{0}    $ {1}{2}".format(indent, command, os.linesep))
    for line in stdout:
        line = _homogenize_linesep(line)
        if line.strip():
            fpointer(indent + "    " + line.replace("\t", "    ") + os.linesep)
        else:
            fpointer(os.linesep)