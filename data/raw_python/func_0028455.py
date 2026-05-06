def delete_custom_view(auth, url, name):
    """
    function takes input of auth, url, and name and issues a RESTFUL call to delete a specific of custom views from HPE
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

    >>> assert view_1 == None

    >>> delete_custom_view(auth.creds, auth.url, name = "L2 View")
    'View L2 View deleted successfully'

    >>> view_2 =get_custom_views( auth.creds, auth.url, name = 'L2 View')

    >>> assert view_2 == None

    """
    view_id  = get_custom_views(auth, url,name )[0]['symbolId']
    delete_custom_view_url = '/imcrs/plat/res/view/custom/'+str(view_id)
    f_url = url + delete_custom_view_url
    r = requests.delete(f_url, auth=auth, headers=HEADERS)  # creates the URL using the payload variable as the contents
    try:
        if r.status_code == 204:
            return 'View ' + name +' deleted successfully'
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + ' delete_custom_view: An Error has occured'