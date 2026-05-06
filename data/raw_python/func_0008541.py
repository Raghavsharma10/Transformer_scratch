def rst2markdown_github(path_to_rst, path_to_md, pandoc="pandoc"):
    """
    Converts ``rst`` to **markdown_github**, using :program:`pandoc`

    **Input**

        * ``FILE.rst``

    **Output**

        * ``FILE.md``

    """
    _proc = subprocess.Popen([pandoc, "-f", "rst",
                              "-t", "markdown_github",
                              #"-o", path_to_md,
                              path_to_rst],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    print("Converting README.rst to markdown_github, "
          "using 'pandoc' ...")
    _stdout, _stderr = _proc.communicate()
    with _open(path_to_md, "w", encoding="utf-8") as _md:
        _skip_newline = False
        for line in _stdout.decode('utf-8').split(os.linesep):
            line = re.sub("``` sourceCode", "``` python", line)
            if line.startswith("[!["):
                _md.write(line)
                _md.write("\n")
                if not line.startswith(("[![LICENSE")):
                    _skip_newline = True
            elif _skip_newline and line == "":
                _skip_newline = False
                continue
            else:
                _md.write(line)
                _md.write("\n")

    if _stderr:
        print("pandoc.exe STDERR: ", _stderr)
    if os.path.isfile(path_to_md) and os.stat(path_to_md).st_size > 0:
        print("README.rst converted and saved as: {}".format(path_to_md))