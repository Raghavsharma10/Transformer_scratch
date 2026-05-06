def is_dependant(package, project_name):
    """Determine whether `package` is a dependant of `project_name`."""
    for requirement in package.requires():
        # perform case-insensitive matching
        if requirement.project_name.lower() == project_name.lower():
            return True
    return False