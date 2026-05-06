def create_temp_file(file_name=None, string_or_another_file=""):
    """
    Creates a temp file using a given name.  Temp files are placed in the Project/temp/ 
    directory.  Any temp files being created with an existing temp file, will be 
    overridden.  This is useful for testing uploads, where you would want to create a 
    temporary file with a desired name, upload it, then delete the file when you're 
    done.

    Kwargs:
        file_name (str): Name of file
        string_or_another_file: Contents to set this file to. If this is set to a file, 
                                it will copy that file.  If this is set to a string, then 
                                it will write this string to the temp file.

    Return: 
        str - Returns the file path to the generated temp file.

    Usage::

        temp_file_path = create_temp_file("mytestfile", "The nimble fox jumps over the lazy dog.")
        file_obj = open(temp_file_path)
        os.remove(temp_file_path)

    """
    temp_file_path = temp_path(file_name)
    if isinstance(string_or_another_file, file):
        # attempt to read it as a file.
        temp_file = open(temp_file_path, "wb")
        temp_file.write(string_or_another_file.read())
    else:
        # handle as a string type if we can't handle as a file.
        temp_file = codecs.open(temp_file_path, "w+", "utf-8")
        temp_file.write(string_or_another_file)

    temp_file.close()
    return temp_file_path