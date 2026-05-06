def find_files(directory=".", ext=None, name=None,
               match_case=False, disable_glob=False, depth=None,
               abspath=False, enable_scandir=False):
    """
    Walk through a file directory and return an iterator of files
    that match requirements. Will autodetect if name has glob as magic
    characters.

    Note: For the example below, you can use find_files_list to return as a
    list, this is simply an easy way to show the output.

    .. code:: python

        list(reusables.find_files(name="ex", match_case=True))
        # ['C:\\example.pdf',
        #  'C:\\My_exam_score.txt']

        list(reusables.find_files(name="*free*"))
        # ['C:\\my_stuff\\Freedom_fight.pdf']

        list(reusables.find_files(ext=".pdf"))
        # ['C:\\Example.pdf',
        #  'C:\\how_to_program.pdf',
        #  'C:\\Hunks_and_Chicks.pdf']

        list(reusables.find_files(name="*chris*"))
        # ['C:\\Christmas_card.docx',
        #  'C:\\chris_stuff.zip']

    :param directory: Top location to recursively search for matching files
    :param ext: Extensions of the file you are looking for
    :param name: Part of the file name
    :param match_case: If name or ext has to be a direct match or not
    :param disable_glob: Do not look for globable names or use glob magic check
    :param depth: How many directories down to search
    :param abspath: Return files with their absolute paths
    :param enable_scandir: on python < 3.5 enable external scandir package
    :return: generator of all files in the specified directory
    """
    if ext or not name:
        disable_glob = True
    if not disable_glob:
        disable_glob = not glob.has_magic(name)

    if ext and isinstance(ext, str):
        ext = [ext]
    elif ext and not isinstance(ext, (list, tuple)):
        raise TypeError("extension must be either one extension or a list")
    if abspath:
        directory = os.path.abspath(directory)
    starting_depth = directory.count(os.sep)

    for root, dirs, files in _walk(directory, enable_scandir=enable_scandir):
        if depth and root.count(os.sep) - starting_depth >= depth:
            continue

        if not disable_glob:
            if match_case:
                raise ValueError("Cannot use glob and match case, please "
                                 "either disable glob or not set match_case")
            glob_generator = glob.iglob(os.path.join(root, name))
            for item in glob_generator:
                yield item
            continue

        for file_name in files:
            if ext:
                for end in ext:
                    if file_name.lower().endswith(end.lower() if not
                    match_case else end):
                        break
                else:
                    continue
            if name:
                if match_case and name not in file_name:
                    continue
                elif name.lower() not in file_name.lower():
                    continue
            yield os.path.join(root, file_name)