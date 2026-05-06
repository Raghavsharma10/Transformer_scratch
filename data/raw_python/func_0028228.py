def add_perf_task(task, auth, url):
    """
    function takes the a python dict containing all necessary fields for a performance tasks,
    transforms the dict into JSON and issues a RESTFUL call to create the performance task. device.

    :param task: dictionary containing all required fields for performance tasks

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: 204

    :rtype: str

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.perf import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> new_task = {'indexDesc': '1.3.6.1.4.1.9.9.13.1.3.1.3',
                    'indexType': '[index1[0]:ciscoEnvMonTemperatureStatusValue:1:0]',
                    'itemFunction': '1.3.6.1.4.1.9.9.13.1.3.1.3',
                    'itemName': 'Cisco_Temperature',
                    'selectDefaultUnit': '400',
                    'unit': 'Celsius'}

    >>> new_perf_task = add_perf_task(new_task, auth.creds, auth.url)
    """
    add_perf_task_url = "/imcrs/perf/task"
    f_url = url + add_perf_task_url
    payload = json.dumps(task)
    response = requests.post(f_url, data=payload, auth=auth, headers=HEADERS)
    try:
        return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' add_perf_task: An Error has occured'