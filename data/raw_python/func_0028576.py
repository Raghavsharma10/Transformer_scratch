def run_dev_cmd(devid, cmd_list, auth, url):
    '''
    Function takes devid of target device and a sequential list of strings which define the specific commands to be run
    on the target device and returns a str object containing the output of the commands.
    :param devid: int devid of the target device

    :param cmd_list: list of strings

    :return: str containing the response of the commands

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.device import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> cmd_list = ['display version']

    >>> cmd_output = run_dev_cmd('10', cmd_list, auth.creds, auth.url)

    >>> assert type(cmd_output) is dict

    >>> assert 'cmdlist' in cmd_output

    >>> assert 'success' in cmd_output


    '''
    run_dev_cmd_url = '/imcrs/icc/confFile/executeCmd'
    f_url = url + run_dev_cmd_url
    cmd_list = _make_cmd_list(cmd_list)
    payload = '''{ "deviceId" : "'''+str(devid) + '''",
                   "cmdlist" : { "cmd" :
                   ['''+ cmd_list + ''']

                   }
                   }'''
    r = requests.post(f_url, data=payload, auth=auth, headers=HEADERS)
    if r.status_code == 200:
        if len(r.text) < 1:
            return ''
        else:
            return json.loads(r.text)