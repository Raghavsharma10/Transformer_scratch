def get_parents():
    """Return sorted list of names of packages without dependants."""
    distributions = get_installed_distributions(user_only=ENABLE_USER_SITE)
    remaining = {d.project_name.lower() for d in distributions}
    requirements = {r.project_name.lower() for d in distributions for
                    r in d.requires()}

    return get_realnames(remaining - requirements)