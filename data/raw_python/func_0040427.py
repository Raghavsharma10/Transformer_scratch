def execute(prog_name, args=None):
    """
    Adapted `compilemessages <http://bit.ly/1r3glSu>`_ command from Django.
    """
    args = _get_parser().parse_args(args or [])
    locale, locale_dir = args.locale, args.locale_dir

    program = 'msgfmt'
    ensure_programs(program)

    def has_bom(fn):
        with open(fn, 'rb') as f:
            sample = f.read(4)
        return (sample[:3] == b'\xef\xbb\xbf'
                or sample.startswith(codecs.BOM_UTF16_LE)
                or sample.startswith(codecs.BOM_UTF16_BE))

    if locale:
        dirs = [os.path.join(locale_dir, l, 'LC_MESSAGES') for l in locale]
    else:
        dirs = [locale_dir, ]
    for ldir in dirs:
        for dir_path, dir_names, file_names in os.walk(ldir):
            for file_name in file_names:
                if not file_name.endswith('.po'):
                    continue
                print_out("Processing file '{:}' in {:}".format(file_name,
                                                                dir_path))
                file_path = os.path.join(dir_path, file_name)
                if has_bom(file_path):
                    raise RuntimeError(
                        "The '{:}' file has a BOM (Byte Order Mark). "
                        "Verboselib supports only .po files encoded in UTF-8 "
                        "and without any BOM.".format(file_path))
                prefix = os.path.splitext(file_path)[0]
                args = [
                    program,
                    '--check-format',
                    '-o',
                    native_path(prefix + '.mo'),
                    native_path(prefix + '.po'),
                ]
                output, errors, status = popen_wrapper(args)
                if status:
                    if errors:
                        msg = "Execution of %s failed: %s" % (program, errors)
                    else:
                        msg = "Execution of %s failed" % program
                    raise RuntimeError(msg)