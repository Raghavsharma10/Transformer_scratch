def get_cfg_template(auth, url, folder=None):
    """
    Function takes no input and returns a list of dictionaries containing the configuration
    templates in the root folder of the icc configuration template library.

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param folder: optional str of name of target folder

    :folder = str of target folder name

    :return: List of Dictionaries containing folders and configuration files in the ICC library.

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.icc import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> config_templates = get_cfg_template(auth.creds, auth.url)

    >>> assert type(config_templates) is list

    >>> assert 'confFileName' in config_templates[0]


    >>> config_templates_folder = get_cfg_template(auth.creds, auth.url, folder='ADP_Configs')

    >>> assert type(config_templates_folder) is list

    >>> assert 'confFileName' in config_templates_folder[0]

    >>> config_template_no_folder = get_cfg_template(auth.creds, auth.url, folder='Doesnt_Exist')

    >>> assert config_template_no_folder is None
    """
    if folder is None:
        get_cfg_template_url = "/imcrs/icc/confFile/list"
    else:
        folder_id = get_folder_id(folder, auth, url)
        get_cfg_template_url = "/imcrs/icc/confFile/list/" + str(folder_id)
    f_url = url + get_cfg_template_url
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            cfg_template_list = (json.loads(response.text))['confFile']
            if isinstance(cfg_template_list, list):
                return cfg_template_list
            elif isinstance(cfg_template_list, dict):
                return [cfg_template_list]
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_cfg_template: An Error has occured"