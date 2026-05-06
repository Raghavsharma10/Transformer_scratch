def execute_by_options(args):
    """execute by argument dictionary

    Args:
        args (dict): command line argument dictionary

    """
    if args['subcommand'] == 'sphinx':
        s = Sphinx(proj_info)
        if args['quickstart']:
            s.quickstart()
        elif args['gen_code_api']:
            s.gen_code_api()
        elif args['rst2html']:
            s.rst2html()
        pass
    elif args['subcommand'] == 'offline_dist':
        pod = PyOfflineDist()
        if args['freeze_deps']:
            pod.freeze_deps()
        elif args['download_deps']:
            pod.download_deps()
        elif args['install_deps']:
            pod.install_deps()
        elif args['clean_deps']:
            pod.clean_deps()
        elif args['mkbinary']:
            pod.pyinstaller_mkbinary(args['mkbinary'])
        elif args['clean_binary']:
            pod.clean_binary()

    pass