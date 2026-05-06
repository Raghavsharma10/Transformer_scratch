def get_custom_views(auth, url, name=None):
    """
    function requires no input and returns a list of dictionaries of custom views from an HPE
    IMC. Optional name argument will return only the specified view.
    :param name: str containing the name of the desired custom view

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param name: (optional) str of name of specific custom view

    :return: list of dictionaties containing attributes of the custom views

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.groups import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> all_views = get_custom_views(auth.creds, auth.url)

    >>> assert type(all_views) is list

    >>> assert 'name' in all_views[0]

    >>> non_existant_view = get_custom_views(auth.creds, auth.url, name = '''Doesn't Exist''')

    >>> assert non_existant_view is None


    """
    get_custom_view_url = None
    if name is None:
        get_custom_view_url = '/imcrs/plat/res/view/custom?resPrivilegeFilter=false&desc=false' \
                               '&total=false'
    elif name is not None:
        get_custom_view_url = '/imcrs/plat/res/view/custom?resPrivilegeFilter=false&name=' + \
                               name + '&desc=false&total=false'
    f_url = url + get_custom_view_url
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            custom_view_list = (json.loads(response.text))
            if 'customView' in custom_view_list:
                custom_view_list = custom_view_list['customView']
                if isinstance(custom_view_list, dict):
                    custom_view_list = [custom_view_list]
                    return custom_view_list
                else:
                    return custom_view_list
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' get_custom_views: An Error has occured'