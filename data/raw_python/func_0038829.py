def get_dependants(project_name):
    """Yield dependants of `project_name`."""
    for package in get_installed_distributions(user_only=ENABLE_USER_SITE):
        if is_dependant(package, project_name):
            yield package.project_name