def get_folder_id(folder_name, auth, url):
    """
    Helper function takes str input of folder name and returns str numerical id of the folder.
    :param folder_name: str name of the folder

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: str numerical id of the folder

    :rtype: str

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.icc import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> default_folder_id = get_folder_id('Default Folder', auth.creds, auth.url)

    >>> assert type(default_folder_id) is str

    """
    object_list = get_cfg_template(auth=auth, url=url)
    for object in object_list:
        if object['confFileName'] == folder_name:
            return object['confFileId']
    return "Folder not found"