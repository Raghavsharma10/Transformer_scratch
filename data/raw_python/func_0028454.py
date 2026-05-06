def add_devs_custom_views(custom_view_name, dev_list, auth, url):
    """
    function takes a list of devIDs from devices discovered in the HPE IMC platform and and issues a RESTFUL call to
     add the list of devices to a specific custom views from HPE IMC.

    :param dev_list: list containing the devID of all devices to be contained in this custom view.

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: str of creation results ( "view " + name + "created successfully"

    :rtype: str

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.groups import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    """
    view_id = get_custom_views(auth, url, name=custom_view_name)[0]['symbolId']
    add_devs_custom_views_url = '/imcrs/plat/res/view/custom/'+str(view_id)
    payload = '''{"device" : '''+ json.dumps(dev_list) + '''}'''
    f_url = url + add_devs_custom_views_url
    r = requests.put(f_url, data = payload, auth=auth, headers=HEADERS)  # creates the URL using the payload variable as the contents
    try:
        if r.status_code == 204:
            print ('View ' + custom_view_name +' : Devices Successfully Added')
            return r.status_code
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + ' get_custom_views: An Error has occured'