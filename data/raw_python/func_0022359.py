def download(name, course, github='SheffieldML/notebook/master/lab_classes/'):
    """Download a lab class from the relevant course
    :param course: the course short name to download the class from.
    :type course: string
    :param reference: reference to the course for downloading the class.
    :type reference: string
    :param github: github repo for downloading the course from.
    :type string: github repo for downloading the lab."""

    github_stub = 'https://raw.githubusercontent.com/'
    if not name.endswith('.ipynb'):
        name += '.ipynb'
    from pods.util import download_url
    download_url(os.path.join(github_stub, github, course, name), store_directory=course)