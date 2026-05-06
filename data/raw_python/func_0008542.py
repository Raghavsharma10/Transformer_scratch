def console_help2rst(cwd, help_cmd, path_to_rst, rst_title,
                     format_as_code=False):
    """
    Extract HELP information from ``<program> -h | --help`` message

    **Input**

        * ``$ <program> -h | --help``
        * ``$ cd <cwd> && make help``

    **Output**

        * ``docs/src/console_help_xy.rst``

    """
    generated_time_str = """

    ::

     generated: {0}

""".format(time.strftime("%d %B %Y - %H:%M"))

    with _open(path_to_rst, "w", encoding='utf-8') as f:
        print("File", f)
        print("cwd", cwd)
        print("help_cmd", help_cmd)
        os.chdir(cwd)
        _proc = subprocess.Popen(
            help_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,)
        help_msg = _proc.stdout.readlines()
        f.write(get_rst_title(
                rst_title,
                "-",
                overline=True))
        f.write(generated_time_str)
        if "README" in path_to_rst:
            help_msg = "".join(help_msg[10:])
            #help_msg = PACKAGE_DOCSTRING + help_msg
        for line in help_msg:
            # exclude directory walk messages of 'make'
            if line.strip().startswith("make[1]:"):
                print("skipped line: {}".format(line))
            # exclude warning messages
            elif line.strip().startswith("\x1b[1m"):
                print("skipped line: {}".format(line))
            # exclude warning messages on Windows (without ``colorama``)
            elif line.strip().startswith("Using fallback version of '"):
                print("skipped line: {}".format(line))
            else:
                # correctly indent tips in 'make help'
                if line.strip().startswith("-->"):
                    f.write(3 * "\t")
                if format_as_code:
                    f.write("\t" + line.strip())
                    f.write("\n")
                else:
                    f.write(line)

        f.write("\n")
        if "README" in path_to_rst:
            f.write(get_rst_title("Credits", "^"))
            f.write(get_credits())

    print("\ncmd:{} in dir:{} --> RST generated:\n\t{}\n\n".format(
        help_cmd, cwd, path_to_rst))