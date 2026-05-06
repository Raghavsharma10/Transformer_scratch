def lint():
    "report pylint results"
    # report according to file extension
    report_formats = {
        ".html": "html",
        ".log": "parseable",
        ".txt": "text",
    }

    lint_build_dir = easy.path("build/lint")
    lint_build_dir.exists() or lint_build_dir.makedirs()  # pylint: disable=expression-not-assigned

    argv = []
    rcfile = easy.options.lint.get("rcfile")
    if not rcfile and easy.path("pylint.cfg").exists():
        rcfile = "pylint.cfg"
    if rcfile:
        argv += ["--rcfile", os.path.abspath(rcfile)]
    if easy.options.lint.get("msg_only", False):
        argv += ["-rn"]
    argv += [
        "--import-graph", (lint_build_dir / "imports.dot").abspath(),
    ]
    argv += support.toplevel_packages()

    sys.stderr.write("Running %s::pylint '%s'\n" % (sys.argv[0], "' '".join(argv)))
    outfile = easy.options.lint.get("output", None)
    if outfile:
        outfile = os.path.abspath(outfile)

    try:
        with easy.pushd("src" if easy.path("src").exists() else "."):
            if outfile:
                argv.extend(["-f", report_formats.get(easy.path(outfile).ext, "text")])
                sys.stderr.write("Writing output to %r\n" % (str(outfile),))
                outhandle = open(outfile, "w")
                try:
                    subprocess.check_call(["pylint"] + argv, stdout=outhandle)
                finally:
                    outhandle.close()
            else:
                subprocess.check_call(["pylint"] + argv, )
            sys.stderr.write("paver::lint - No problems found.\n")
    except subprocess.CalledProcessError as exc:
        if exc.returncode & 32:
            # usage error (internal error in this code)
            sys.stderr.write("paver::lint - Usage error, bad arguments %r?!\n" % (argv,))
            sys.exit(exc.returncode)
        else:
            bits = {
                1: "fatal",
                2: "error",
                4: "warning",
                8: "refactor",
                16: "convention",
            }
            sys.stderr.write("paver::lint - Some %s message(s) issued.\n" % (
                ", ".join([text for bit, text in bits.items() if exc.returncode & bit])
            ))
            if exc.returncode & 3:
                sys.stderr.write("paver::lint - Exiting due to fatal / error message.\n")
                sys.exit(exc.returncode)