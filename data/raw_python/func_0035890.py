def resource_copy(package_or_requirement, resource_name, destination):
    '''
    Copy file/dir resource to destination.

    Parameters
    ----------
    package_or_requirement : str
    resource_name : str
    destination : ~pathlib.Path
        Path to copy to, it must not exist.
    '''
    args = package_or_requirement, resource_name
    if resource_isdir(*args):
        destination.mkdir()
        for name in resource_listdir(*args):
            resource_copy(
                package_or_requirement,
                str(Path(resource_name) / name),
                destination / name
            )
    else:
        with destination.open('wb') as f:
            with resource_stream(*args) as source:
                shutil.copyfileobj(source, f)