def download_to_tempfile(url, file_name=None, extension=None):
    """
    Downloads a URL contents to a tempfile.  This is useful for testing downloads.
    It will download the contents of a URL to a tempfile, which you then can 
    open and use to validate the downloaded contents.

    Args:
        url (str) : URL of the contents to download.

    Kwargs:
        file_name (str): Name of file.
        extension (str): Extension to use.

    Return:
        str - Returns path to the temp file.

    """

    if not file_name:
        file_name = generate_timestamped_string("wtf_temp_file")

    if extension:
        file_path = temp_path(file_name + extension)
    else:
        ext = ""
        try:
            ext = re.search(u"\\.\\w+$", file_name).group(0)
        except:
            pass
        file_path = temp_path(file_name + ext)

    webFile = urllib.urlopen(url)
    localFile = open(file_path, 'w')
    localFile.write(webFile.read())
    webFile.close()
    localFile.close()

    return file_path