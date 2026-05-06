def run_main(args: argparse.Namespace, do_exit=True) -> None:
    """Runs the checks and exits.

    To extend this tool, use this function and set do_exit to False
    to get returned the status code.
    """
    if args.init:
        generate()
        return None  # exit after generate instead of starting to lint

    handler = CheckHandler(
        file=args.config_file, out_json=args.json, files=args.files)

    for style in get_stylers():
        handler.run_linter(style())

    for linter in get_linters():
        handler.run_linter(linter())

    for security in get_security():
        handler.run_linter(security())

    for tool in get_tools():
        tool = tool()

        # Only run pypi if everything else passed
        if tool.name == "pypi" and handler.status_code != 0:
            continue

        handler.run_linter(tool)

    if do_exit:
        handler.exit()
    return handler.status_code