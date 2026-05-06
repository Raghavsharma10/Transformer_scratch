def unzip_file(zip_fname):
    """Unzip the zip_fname in the current directory.""" 
    print("Unzipping {}".format(zip_fname))
    with zipfile.ZipFile(zip_fname) as zf:
        zf.extractall()