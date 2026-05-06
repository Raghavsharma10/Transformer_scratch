def _main():
    "ctox: tox with conda"
    from sys import argv
    arguments = argv[1:]

    toxinidir = os.getcwd()

    return main(arguments, toxinidir)