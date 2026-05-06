def add_devs_custom_views(custom_view_name, dev_list, auth, url):
    """
    function takes a list of devIDs from devices discovered in the HPE IMC platform and issues a
    RESTFUL call to add the list of devices to a specific custom views from HPE IMC.

    :param custom_view_name: str of the target custom view name

    :param dev_list: list containing the devID of all devices to be contained in this custom view.

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: str of creation results ( "view " + name + "created successfully"

    :rtype: str

    >>> from pyhpeimc.auth import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    """
    view_id = get_custom_views(auth, url, name=custom_view_name)
    if view_id is None:
        print("View " + custom_view_name + " doesn't exist")
        return view_id
    view_id = get_custom_views(auth, url, name=custom_view_name)[0]['symbolId']
    add_devs_custom_views_url = '/imcrs/plat/res/view/custom/' + str(view_id)
    device_list = []
    for dev in dev_list:
        new_dev = {"id": dev}
        device_list.append(new_dev)
    payload = '''{"device" : ''' + json.dumps(device_list) + '''}'''
    print(payload)
    f_url = url + add_devs_custom_views_url
    response = requests.put(f_url, data=payload, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 204:
            print('View ' + custom_view_name + ' : Devices Successfully Added')
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' get_custom_views: An Error has occured'