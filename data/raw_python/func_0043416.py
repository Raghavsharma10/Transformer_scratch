def run_command(cmd, root, dst, input, params):
    """
    Execute a command, and if successful write it's stdout to ``root``/``dst``.
    """
    use_stdout = '{output}' not in cmd
    if not use_stdout:
        params['output'] = dst
    parsed_cmd = parse_command(cmd, input=input, params=params)

    ensure_dirs(os.path.join(root, dst))

    logger.info("Running [%s] from [%s]", parsed_cmd, root)

    proc = subprocess.Popen(
        args=parsed_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        cwd=root,
    )
    (stdout, stderr) = proc.communicate()

    assert not proc.returncode, stderr

    if use_stdout:
        # TODO: this should probably change dest to be a temp file
        with open(os.path.join(root, dst), 'w') as fp:
            fp.write(stdout)