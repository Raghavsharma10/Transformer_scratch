def project2module(project):
    """Convert project name into a module name."""
    # Name unification in accordance with PEP 426.
    project = project.lower().replace("-", "_")
    if project.startswith("python_"):
        # Remove conventional "python-" prefix.
        project = project[7:]
    return project