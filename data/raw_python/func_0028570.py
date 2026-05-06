def get_system_device_models(auth, url):
    """Takes string no input to issue RESTUL call to HP IMC\n

      :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

      :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

      :return: list of dictionaries where each dictionary represents a single device model

      :rtype: list

      >>> from pyhpeimc.auth import *

      >>> from pyhpeimc.plat.device import *

      >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

      >>> device_models = get_system_device_models(auth.creds, auth.url)

      >>> assert type(device_models) is list

      >>> assert 'virtualDeviceName' in device_models[0]

    """
    get_system_device_model_url = '/imcrs/plat/res/model?start=0&size=10000&orderBy=id&desc=false&total=false'
    f_url = url + get_system_device_model_url
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=HEADERS)
    # r.status_code
    try:
        if r.status_code == 200:
            system_device_model = (json.loads(r.text))
            return system_device_model['deviceModel']
    except requests.exceptions.RequestException as e:
        return "Error:\n" + str(e) + " get_dev_details: An Error has occured"