def get_package_release_from_pypi(pkg_name, version, pypi_json_api_url, allowed_classifiers):
    """
    No classifier-based selection of Python packages is currently implemented: for now we don't fetch any .whl or .egg
    Eventually, we should select the best release available, based on the classifier & PEP 425: https://www.python.org/dev/peps/pep-0425/
    E.g. a wheel when available but NOT for tornado 4.3 for example, where available wheels are only for Windows.
    Note also that some packages don't have .whl distributed, e.g. https://bugs.launchpad.net/lxml/+bug/1176147
    """
    matching_releases = get_package_releases_matching_version(pkg_name, version, pypi_json_api_url)
    src_releases = [release for release in matching_releases if release['python_version'] == 'source']
    if src_releases:
        return select_src_release(src_releases, pkg_name, target_classifiers=('py2.py3-none-any',), select_arbitrary_version_if_none_match=True)
    if allowed_classifiers:
        return select_src_release(matching_releases, pkg_name, target_classifiers=allowed_classifiers)
    raise PypiQueryError('No source supported found for package {} version {}'.format(pkg_name, version))