def get_custom_view_details(name, auth, url):
    """
    function requires no input and returns a list of dictionaries of custom views from an HPE IMC. Optional name
    argument will return only the specified view.
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
    view_id = get_custom_views(auth, url, name=name)[0]['symbolId']
    get_custom_view_details_url = '/imcrs/plat/res/view/custom/' + str(view_id)
    f_url = url + get_custom_view_details_url
    r = requests.get(f_url, auth=auth,
                     headers=HEADERS)  # creates the URL using the payload variable as the contents
    try:
        if r.status_code == 200:
            current_devices = (json.loads(r.text))
            if 'device' in current_devices:
                return current_devices['device']
            else:
                return []
    except requests.exceptions.RequestException as e:
        return "Error:\n" + str(e) + ' get_custom_views: An Error has occured'