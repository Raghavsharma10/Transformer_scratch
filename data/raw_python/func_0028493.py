def get_vm_host_info(hostId):
    """
    function takes hostId as input to RESTFUL call to HP IMC
    :param hostId: int or string of HostId of Hypervisor host
    :return:list of dictionatires contraining the VM Host information for the target hypervisor
    """
    global r
    if auth is None or url is None:  # checks to see if the imc credentials are already available
        set_imc_creds()
    get_vm_host_info_url = "/imcrs/vrm/host?hostId=" + str(hostId)
    f_url = url + get_vm_host_info_url
    payload = None
    r = requests.get(f_url, auth=auth,
                     headers=headers)  # creates the URL using the payload variable as the contents
    # print(r.status_code)
    if r.status_code == 200:
        if len(r.text) > 0:
            return json.loads(r.text)
    elif r.status_code == 204:
        print("Device is not a supported Hypervisor")
        return "Device is not a supported Hypervisor"
    else:
        print("An Error has occured")