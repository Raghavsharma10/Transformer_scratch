def delete_custom_view(auth, url, name):
    """
    function takes input of auth, url, and name and issues a RESTFUL call to delete a specific
    of custom views from HPE
    IMC.
    :param name: string containg the name of the desired custom view

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: str of creation results ( "view " + name + "created successfully"

    :rtype: str

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.groups import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> delete_custom_view(auth.creds, auth.url, name = "L1 View")
    'View L1 View deleted successfully'

    >>> view_1 =get_custom_views( auth.creds, auth.url, name = 'L1 View')

    >>> assert view_1 is None

    >>> delete_custom_view(auth.creds, auth.url, name = "L2 View")
    'View L2 View deleted successfully'

    >>> view_2 =get_custom_views( auth.creds, auth.url, name = 'L2 View')

    >>> assert view_2 is None

    """
    view_id = get_custom_views(auth, url, name)
    if view_id is None:
        print("View " + name + " doesn't exists")
        return view_id
    view_id = get_custom_views(auth, url, name)[0]['symbolId']
    delete_custom_view_url = '/imcrs/plat/res/view/custom/' + str(view_id)
    f_url = url + delete_custom_view_url
    response = requests.delete(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 204:
            print('View ' + name + ' deleted successfully')
            return response.status_code
        else:
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' delete_custom_view: An Error has occured'