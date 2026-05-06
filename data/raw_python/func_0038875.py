def find_owners(path):
    """Return the package(s) that file belongs to."""
    abspath = os.path.abspath(path)

    packages = search_packages_info(
        sorted((d.project_name for d in
                get_installed_distributions(user_only=ENABLE_USER_SITE)),
               key=lambda d: d.lower()))

    return [p['name'] for p in packages if is_owner(p, abspath)]