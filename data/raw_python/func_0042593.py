def performInDirectory(dirPath):
    """
    Change the current working directory to dirPath before performing
    an operation, then restore the original working directory after
    """
    originalDirectoryPath = os.getcwd()
    try:
        os.chdir(dirPath)
        yield
    finally:
        os.chdir(originalDirectoryPath)