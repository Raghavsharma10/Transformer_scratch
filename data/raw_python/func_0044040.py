def main(args=None):
    """Command line interface.

    :param list args: command line options (defaults to sys.argv)
    :returns: exit code
    :rtype: int

    """
    parser = ArgumentParser(
        prog='baseline',
        description=DESCRIPTION)

    parser.add_argument(
        'path', nargs='*',
        help='module or directory path')

    parser.add_argument(
        '--movepath', help='location to move script updates')

    parser.add_argument(
        '-w', '--walk', action='store_true',
        help='recursively walk directories')

    args = parser.parse_args(args)

    paths = args.path or ['.']

    paths = [path for pattern in paths for path in glob(pattern)]

    if args.walk:
        for dirpath in (p for p in paths if os.path.isdir(p)):
            for root, _dirs, files in os.walk(dirpath):
                paths += (os.path.join(root, filename) for filename in files)
    else:
        for dirpath in (p for p in paths if os.path.isdir(p)):
            paths += (os.path.join(dirpath, pth) for pth in os.listdir(dirpath))

    update_paths = [
        os.path.relpath(p) for p in paths if p.lower().endswith(UPDATE_EXT)]

    exitcode = 0

    if update_paths:
        script_paths = [pth[:-len(UPDATE_EXT)] + '.py' for pth in update_paths]

        print('Found updates for:')
        for path in script_paths:
            print('  ' + path)
        print()

        if not args.movepath:
            try:
                input('Hit [ENTER] to update, [Ctrl-C] to cancel ')
            except KeyboardInterrupt:
                print()
                print('Update canceled.')
                exitcode = 1
            else:
                print()

        if exitcode == 0:
            for script_path, update_path in zip(script_paths, update_paths):
                if args.movepath:
                    script_path = os.path.join(args.movepath, script_path)
                    if update_path.startswith('..'):
                        raise RuntimeError(
                            'destination outside of move path: ' + script_path)
                    script_dirpath = os.path.dirname(script_path)
                    if not os.path.isdir(script_dirpath):
                        os.makedirs(script_dirpath)
                with open(update_path) as update:
                    new_content = update.read()
                with open(script_path, 'w') as script:
                    script.write(new_content)
                os.remove(update_path)
                print(update_path + ' -> ' + script_path)

    return exitcode