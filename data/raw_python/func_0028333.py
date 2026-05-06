def get_vm_host_info(hostip, auth, url):
    """
    function takes hostId as input to RESTFUL call to HP IMC

    :param hostip: int or string of hostip of Hypervisor host

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: Dictionary contraining the information for the target VM host

    :rtype: dict

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.vrm import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> host_info = get_vm_host_info('10.101.0.6', auth.creds, auth.url)

    >>> assert type(host_info) is dict

    >>> assert len(host_info) == 10

    >>> assert 'cpuFeg' in host_info

    >>> assert 'cpuNum' in host_info

    >>> assert 'devId' in host_info

    >>> assert 'devIp' in host_info

    >>> assert 'diskSize' in host_info

    >>> assert 'memory' in host_info

    >>> assert 'parentDevId' in host_info

    >>> assert 'porductFlag' in host_info

    >>> assert 'serverName' in host_info

    >>> assert 'vendor' in host_info

    """
    hostId = get_dev_details(hostip, auth, url)['id']
    get_vm_host_info_url = "/imcrs/vrm/host?hostId=" + str(hostId)
    f_url = url + get_vm_host_info_url
    payload = None
    r = requests.get(f_url, auth=auth,
                     headers=HEADERS)  # creates the URL using the payload variable as the contents
    # print(r.status_code)
    try:
        if r.status_code == 200:
            if len(r.text) > 0:
                return json.loads(r.text)
        elif r.status_code == 204:
            print("Device is not a supported Hypervisor")
            return "Device is not a supported Hypervisor"
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " get_vm_host_info: An Error has occured"