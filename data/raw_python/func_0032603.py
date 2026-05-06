def packexe(exefile, srcdir):
    """Pack the files in srcdir into exefile using 7z.

    Requires that stub files are available in checkouts/stubs"""
    exefile = cygpath(os.path.abspath(exefile))
    appbundle = exefile + ".app.7z"

    # Make sure that appbundle doesn't already exist
    # We don't want to risk appending to an existing file
    if os.path.exists(appbundle):
        raise OSError("%s already exists" % appbundle)

    files = os.listdir(srcdir)

    SEVENZIP_ARGS = ['-r', '-t7z', '-mx', '-m0=BCJ2', '-m1=LZMA:d27',
                     '-m2=LZMA:d19:mf=bt2', '-m3=LZMA:d19:mf=bt2', '-mb0:1', '-mb0s1:2',
                     '-mb0s2:3', '-m1fb=128', '-m1lc=4']

    # First, compress with 7z
    stdout = tempfile.TemporaryFile()
    try:
        check_call([SEVENZIP, 'a'] + SEVENZIP_ARGS + [appbundle] + files,
                   cwd=srcdir, stdout=stdout, preexec_fn=_noumask)
    except Exception:
        stdout.seek(0)
        data = stdout.read()
        log.error(data)
        log.exception("Error packing exe %s from %s", exefile, srcdir)
        raise
    stdout.close()

    # Then prepend our stubs onto the compressed 7z data
    o = open(exefile, "wb")
    parts = [
        'checkouts/stubs/7z/7zSD.sfx.compressed',
        'checkouts/stubs/tagfile/app.tag',
        appbundle
    ]
    for part in parts:
        i = open(part)
        while True:
            block = i.read(4096)
            if not block:
                break
            o.write(block)
        i.close()
    o.close()
    os.unlink(appbundle)