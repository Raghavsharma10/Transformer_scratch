def DirectoryFixmatFactory(directory, categories = None, glob_str = '*.mat', var_name = 'fixmat'):
    """
    Concatenates all fixmats in dir and returns the resulting single
    fixmat.
    
    Parameters:
        directory : string
            Path from which the fixmats should be loaded
        categories : instance of stimuli.Categories, optional
            If given, the resulting fixmat provides direct access
            to the data in the categories object.
        glob_str : string
            A regular expression that defines which mat files are picked up.
        var_name : string
            The variable to load from the mat file.
    Returns:
        f_all : instance of FixMat
            Contains all fixmats that were found in given directory
        
    """
    files = glob(join(directory,glob_str))
    if len(files) == 0:
        raise ValueError("Could not find any fixmats in " + 
            join(directory, glob_str))
    f_all = FixmatFactory(files.pop(), categories, var_name)
    for fname in files:
        f_current = FixmatFactory(fname, categories, var_name)
        f_all.join(f_current)
    return f_all