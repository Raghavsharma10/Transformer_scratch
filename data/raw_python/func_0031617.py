def run(macro, output_files=[], force_close=True):
    """
    Runs Fiji with the suplied macro. Output of Fiji can be viewed by
    setting environment variable `DEBUG=fijibin`.

    Parameters
    ----------
    macro : string or list of strings
        IJM-macro(s) to run. If list of strings, it will be joined with
        a space, so all statements should end with ``;``.
    output_files : list
        Files to check if exists after macro has been run. Files specified that
        do not exist after macro is done will print a warning message.
    force_close : bool
        Will add ``eval("script", "System.exit(42);");`` to end of macro. Exit
        code 42 is used to overcome that errors in macro efficiently will exit
        Fiji with error code 0. In other words, if this line in the macro is
        reached, the macro has most probably finished without errors. This
        is the default behaviour.

        One should also note that Fiji doesn't terminate right away if
        ``System.exit()`` is left out, and it may take several minutes for
        Fiji to close.

    Returns
    -------
    int
        Files from output_files which exists after running macro.
    """
    if type(macro) == list:
        macro = ' '.join(macro)
    if len(macro) == 0:
        print('fijibin.macro.run got empty macro, not starting fiji')
        return _exists(output_files)
    if force_close:
        # make sure fiji halts immediately when done
        # hack: use error code 42 to check if macro has run sucessfully
        macro = macro + 'eval("script", "System.exit(42);");'

    # escape backslashes (windows file names)
    #                 not \ \  not \      g1 \\ g2
    macro = re.sub(r"([^\\])\\([^\\])", r"\1\\\\\2", macro)

    debug('macro {}'.format(macro))

    # avoid verbose output of Fiji when DEBUG environment variable set
    env = os.environ.copy()
    debugging = False
    if 'DEBUG' in env:
        if env['DEBUG'] == 'fijibin' or env['DEBUG'] == '*':
            debugging = True
        del env['DEBUG']

    fptr, temp_filename = mkstemp(suffix='.ijm')
    m = os.fdopen(fptr, 'w')
    m.write(macro)
    m.flush() # make sure macro is written before running Fiji
    m.close()

    cmd = [fijibin.BIN, '--headless', '-macro', temp_filename]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    out, err = proc.communicate()

    for line in out.decode('latin1', errors='ignore').splitlines():
        debug('stdout:' + line)
    for line in err.decode('latin1', errors='ignore').splitlines():
        debug('stderr:' + line)

    if force_close and proc.returncode != 42:
        print('fijibin ERROR: Fiji did not successfully ' +
              'run macro {}'.format(temp_filename))
        if not debugging:
            print('fijibin Try running script with ' +
                  '`DEBUG=fijibin python your_script.py`')
    else:
        # only delete if everything is ok
        os.remove(temp_filename)


    # return output_files which exists
    return _exists(output_files)