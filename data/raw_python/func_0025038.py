def find_requirements(path):
    """
    This method tries to determine the requirements of a particular project
    by inspecting the possible places that they could be defined.

    It will attempt, in order:

    1) to parse setup.py in the root for an install_requires value
    2) to read a requirements.txt file or a requirements.pip in the root
    3) to read all .txt files in a folder called 'requirements' in the root
    4) to read files matching "*requirements*.txt" and "*reqs*.txt" in the root,
       excluding any starting or ending with 'test'

    If one of these succeeds, then a list of pkg_resources.Requirement's
    will be returned. If none can be found, then a RequirementsNotFound
    will be raised
    """
    requirements = []

    setup_py = os.path.join(path, 'setup.py')
    if os.path.exists(setup_py) and os.path.isfile(setup_py):
        try:
            requirements = from_setup_py(setup_py)
            requirements.sort()
            return requirements
        except CouldNotParseRequirements:
            pass

    for reqfile_name in ('requirements.txt', 'requirements.pip'):
        reqfile_path = os.path.join(path, reqfile_name)
        if os.path.exists(reqfile_path) and os.path.isfile(reqfile_path):
            try:
                requirements += from_requirements_txt(reqfile_path)
            except CouldNotParseRequirements as e:
                pass

    requirements_dir = os.path.join(path, 'requirements')
    if os.path.exists(requirements_dir) and os.path.isdir(requirements_dir):
        from_dir = from_requirements_dir(requirements_dir)
        if from_dir is not None:
            requirements += from_dir

    from_blob = from_requirements_blob(path)
    if from_blob is not None:
        requirements += from_blob

    requirements = list(set(requirements))
    if len(requirements) > 0:
        requirements.sort()
        return requirements

    raise RequirementsNotFound