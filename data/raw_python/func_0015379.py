def do_check(vext_files):
    """
    Attempt to import everything in the 'test-imports' section of specified
    vext_files

    :param: list of vext filenames (without paths), '*' matches all.
    :return: True if test_imports was successful from all files
    """
    import vext
    # not efficient ... but then there shouldn't be many of these

    all_specs = set(vext.gatekeeper.spec_files_flat())
    if vext_files == ['*']:
        vext_files = all_specs
    unknown_specs = set(vext_files) - all_specs
    for fn in unknown_specs:
        print("%s is not an installed vext file." % fn, file=sys.stderr)

    if unknown_specs:
        return False

    check_passed = True
    for fn in [join(vext.gatekeeper.spec_dir(), fn) for fn in vext_files]:
        f = open_spec(open(fn))
        modules = f.get('test_import', [])
        for success, module in vext.gatekeeper.test_imports(modules):
            if not success:
                check_passed = False
            line = "import %s: %s" % (module, '[success]' if success else '[failed]')
            print(line)
        print('')

    return check_passed