def find_version(*file_paths):
    """
    read __init__.py
    """
    file_path = os.path.join(*file_paths)
    with open(file_path, 'r') as version_file:
        line = version_file.readline()
        while line:
            if line.startswith('__version__'):
                version_match = re.search(
                    r"^__version__ = ['\"]([^'\"]*)['\"]",
                    line,
                    re.M
                )
                if version_match:
                    return version_match.group(1)
            line = version_file.readline()
    raise RuntimeError('Unable to find version string.')