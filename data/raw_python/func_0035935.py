def hash(path, hash_function=hashlib.sha512):  # @ReservedAssignment
    '''
    Hash file or directory.

    Parameters
    ----------
    path : ~pathlib.Path
        File or directory to hash.
    hash_function : ~typing.Callable[[], hash object]
        Function which creates a hashlib hash object when called. Defaults to
        ``hashlib.sha512``.

    Returns
    -------
    hash object
        hashlib hash object of file/directory contents. File/directory stat data
        is ignored. The directory digest covers file/directory contents and
        their location relative to the directory being digested. The directory
        name itself is ignored.
    '''
    hash_ = hash_function()
    if path.is_dir():
        for directory, directories, files in os.walk(str(path), topdown=True):
            # Note:
            # - directory: path to current directory in walk relative to current working direcotry
            # - directories/files: dir/file names

            # Note: file names can contain nearly any character (even newlines).

            # hash like (ignore the whitespace):
            #
            #   h(relative-dir-path)
            #   h(dir_name)
            #   h(dir_name2)
            #   ,
            #   h(file_name) h(file_content)
            #   h(file_name2) h(file_content2)
            #   ;
            #   h(relative-dir-path2)
            #   ...
            hash_.update(hash_function(str(Path(directory).relative_to(path)).encode()).digest())
            for name in sorted(directories):
                hash_.update(hash_function(name.encode()).digest())
            hash_.update(b',')
            for name in sorted(files):
                hash_.update(hash_function(name.encode()).digest())
                hash_.update(hash(Path(directory) / name).digest())
            hash_.update(b';')
    else:
        with path.open('rb') as f:
            while True:
                buffer = f.read(65536)
                if not buffer:
                    break
                hash_.update(buffer)
    return hash_