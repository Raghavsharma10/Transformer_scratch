def create_custom_views(auth, url, name=None, upperview=None):
    """
    function takes no input and issues a RESTFUL call to get a list of custom views from HPE IMC.
    Optional Name input will return only the specified view.

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param name: string containg the name of the desired custom view

    :param upperview: str contraining the name of the desired parent custom view

    :return: str of creation results ( "view " + name + "created successfully"

    :rtype: str

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.groups import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    #Create L1 custom view
    >>> create_custom_views(auth.creds, auth.url, name='L1 View')
    'View L1 View created successfully'

    >>> view_1 =get_custom_views( auth.creds, auth.url, name = 'L1 View')

    >>> assert type(view_1) is list

    >>> assert view_1[0]['name'] == 'L1 View'

    #Create Nested custome view
    >>> create_custom_views(auth.creds, auth.url, name='L2 View', upperview='L1 View')
    'View L2 View created successfully'

    >>> view_2 = get_custom_views( auth.creds, auth.url, name = 'L2 View')

    >>> assert type(view_2) is list

    >>> assert view_2[0]['name'] == 'L2 View'

    """
    create_custom_views_url = '/imcrs/plat/res/view/custom?resPrivilegeFilter=false&desc=false' \
                               '&total=false'
    f_url = url + create_custom_views_url
    if upperview is None:
        payload = '''{ "name": "''' + name + '''",
         "upLevelSymbolId" : ""}'''
    else:
        parentviewid = get_custom_views(auth, url, upperview)[0]['symbolId']
        payload = '''{ "name": "''' + name + '''",
        "upLevelSymbolId" : "''' + str(parentviewid) + '''"}'''
    response = requests.post(f_url, data=payload, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 201:
            print('View ' + name + ' created successfully')
            return response.status_code
        elif response.status_code == 409:
            print("View " + name + " already exists")
            return response.status_code
        else:
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' get_custom_views: An Error has occured'