def run_dev_cmd(cmd_list, auth, url, devid=None, devip=None):
    """
    Function takes devid of target device and a sequential list of strings which define the
    specific commands to be run on the target device and returns a str object containing the
    output of the commands.

    :param devid: int devid of the target device

    :param cmd_list: list of strings

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param devip: str of ipv4 address of the target device

    :return: str containing the response of the commands

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.device import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> cmd_list = ['display version']

    >>> cmd_output = run_dev_cmd( cmd_list, auth.creds, auth.url, devid ='10')

    >>> cmd_output = run_dev_cmd( cmd_list, auth.creds, auth.url, devip='10.101.0.221')

    >>> assert type(cmd_output) is dict

    >>> assert 'cmdlist' in cmd_output

    >>> assert 'success' in cmd_output

    """
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    run_dev_cmd_url = '/imcrs/icc/confFile/executeCmd'
    f_url = url + run_dev_cmd_url
    cmd_list = _make_cmd_list(cmd_list)
    payload = '''{ "deviceId" : "''' + str(devid) + '''",
                   "cmdlist" : { "cmd" :
                   [''' + cmd_list + ''']
                   }
                   }'''
    try:
        response = requests.post(f_url, data=payload, auth=auth, headers=HEADERS)
        if response.status_code == 200:
            if len(response.text) < 1:
                return ''
            else:
                return json.loads(response.text)
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " run_dev_cmd: An Error has occured"