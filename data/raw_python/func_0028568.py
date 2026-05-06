def get_system_vendors(auth, url):
    """Takes string no input to issue RESTUL call to HP IMC\n

      :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

      :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

      :return: list of dictionaries where each dictionary represents a single vendor

      :rtype: list

      >>> from pyhpeimc.auth import *

      >>> from pyhpeimc.plat.device import *

      >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

      >>> vendors = get_system_vendors(auth.creds, auth.url)

      >>> assert type(vendors) is list

      >>> assert 'name' in vendors[0]


    """
    get_system_vendors_url = '/imcrs/plat/res/vendor?start=0&size=10000&orderBy=id&desc=false&total=false'
    f_url = url + get_system_vendors_url
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=HEADERS)
    # r.status_code
    try:
        if r.status_code == 200:
            system_vendors = (json.loads(r.text))
            return system_vendors['deviceVendor']
    except requests.exceptions.RequestException as e:
        return "Error:\n" + str(e) + " get_dev_details: An Error has occured"