def launch(exec_, args):
    """
    Launches application.
    """
    if not exec_:
        raise RuntimeError(
            'Mayalauncher could not find a maya executable, please specify'
            'a path in the config file (-e) or add the {} directory location'
            'to your PATH system environment.'.format(DEVELOPER_NAME)
       )

    # Launch Maya
    if args.debug:
        return

    watched = WatchFile()

    cmd = [exec_] if args.file is None else [exec_, args.file]
    cmd.extend(['-hideConsole', '-log', watched.path])
    if args.debug:
        cmd.append('-noAutoloadPlugins')
    maya = subprocess.Popen(cmd)

    # Maya 2016 stupid clic ipm
    # os.environ['MAYA_DISABLE_CLIC_IPM'] = '1'
    # os.environ['MAYA_DISABLE_CIP'] = '1'
    # os.environ['MAYA_OPENCL_IGNORE_DRIVER_VERSION'] = '1'

    while True:
        time.sleep(1)

        maya.poll()
        watched.check()
        if maya.returncode is not None:
            if not maya.returncode == 0:
                maya = subprocess.Popen(cmd)
            else:
                watched.stop()
                break