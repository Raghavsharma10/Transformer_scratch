def get_cfg_template(auth, url, folder = None):
    '''
    Function takes no input and returns a list of dictionaries containing the configuration templates in the root folder
    of the icc configuration template library.

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

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

    >>> assert config_template_no_folder == None
    '''
    if folder == None:
        get_cfg_template_url = "/imcrs/icc/confFile/list"
    else:
        folder_id = get_folder_id(folder, auth, url)
        get_cfg_template_url = "/imcrs/icc/confFile/list/"+str(folder_id)
    f_url = url + get_cfg_template_url
    r = requests.get(f_url,auth=auth, headers=HEADERS)
    #print (r.status_code)
    try:
        if r.status_code == 200:
            cfg_template_list = (json.loads(r.text))
            return cfg_template_list['confFile']

    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " get_cfg_template: An Error has occured"