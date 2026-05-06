def get_template_id(template_name, auth, url):
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

    >>> file_id = get_template_id('CW7SNMP.cfg', auth.creds, auth.url)

    >>> assert type(file_id) is str

    """
    object_list = get_cfg_template(auth=auth, url=url)
    for object in object_list:
        if object['confFileName'] == template_name:
            return object['confFileId']
    return "template not found"