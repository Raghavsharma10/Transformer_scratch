def get_custom_view_details(name, auth, url):
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

    >>> view_details = get_custom_view_details('My Network View', auth.creds, auth.url)

    >>> assert type(view_details) is list

    >>> assert 'label' in view_details[0]

    """
    view_id = get_custom_views(auth, url, name=name)
    if view_id is None:
        return view_id
    view_id = get_custom_views(auth, url, name=name)[0]['symbolId']
    get_custom_view_details_url = '/imcrs/plat/res/view/custom/' + str(view_id)
    f_url = url + get_custom_view_details_url
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            current_devices = (json.loads(response.text))
            if 'device' in current_devices:
                if isinstance(current_devices['device'], dict):
                    return [current_devices['device']]
                else:
                    return current_devices['device']
            else:
                return []
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' get_custom_views: An Error has occured'