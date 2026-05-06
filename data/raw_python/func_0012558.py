def ls(params="", directory=".", printed=True):
    """Know the best python implantation of ls? It's just to subprocess ls...
    (uses dir on windows).

    :param params: options to pass to ls or dir
    :param directory: if not this directory
    :param printed: If you're using this, you probably wanted it just printed
    :return: if not printed, you can parse it yourself
    """
    command = "{0} {1} {2}".format("ls" if not win_based else "dir",
                                   params, directory)
    response = run(command, shell=True)  # Shell required for windows
    response.check_returncode()
    if printed:
        print(response.stdout.decode("utf-8"))
    else:
        return response.stdout