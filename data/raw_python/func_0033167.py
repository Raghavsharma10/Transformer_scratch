def get_tmp_filename(tmp_dir=gettempdir(), prefix="tmp", suffix=".txt",
                     result_constructor=FilePath):
    """ Generate a temporary filename and return as a FilePath object

        tmp_dir: the directory to house the tmp_filename
        prefix: string to append to beginning of filename
            Note: It is very useful to have prefix be descriptive of the
            process which is creating the temporary file. For example, if
            your temp file will be used to build a temporary blast database,
            you might pass prefix=TempBlastDB
        suffix: the suffix to be appended to the temp filename
        result_constructor: the constructor used to build the result filename
            (default: cogent.app.parameters.FilePath). Note that joining
            FilePath objects with one another or with strings, you must use
            the + operator. If this causes trouble, you can pass str as the
            the result_constructor.
    """
    # check not none
    if not tmp_dir:
        tmp_dir = ""
    # if not current directory, append "/" if not already on path
    elif not tmp_dir.endswith("/"):
        tmp_dir += "/"

    chars = "abcdefghigklmnopqrstuvwxyz"
    picks = chars + chars.upper() + "0123456790"
    return result_constructor(tmp_dir) + result_constructor(prefix) +\
        result_constructor("%s%s" %
                           (''.join([choice(picks) for i in range(20)]),
                            suffix))