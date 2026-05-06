def print_build_help(build_path, default_build_path):
    """
    Print help text after configuration step is done.
    """
    print('   configure step is done')
    print('   now you need to compile the sources:')
    if (build_path == default_build_path):
        print('   $ cd build')
    else:
        print('   $ cd ' + build_path)
    print('   $ make')