def command_line_runner():
    """ I run functions from the command-line! """
    filename = sys.argv[-1]
    if not filename.endswith(".rst"):
        print("ERROR! Please enter a ReStructuredText filename!")
        sys.exit()
    print(rst_to_json(file_opener(filename)))