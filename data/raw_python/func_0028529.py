def get_template_id(template_name, auth, url):
    """
    Helper function takes str input of folder name and returns str numerical id of the folder.
    :param template_name: str name of the target template

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: str numerical id of the folder

    :rtype: str

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.icc import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> file_id = get_template_id('CW7SNMP.cfg', auth.creds, auth.url)

    >>> assert type(file_id) is int

    """
    object_list = get_cfg_template(auth=auth, url=url)
    for template in object_list:
        if template['confFileName'] == template_name:
            return int(template['confFileId'])
    return "template not found"