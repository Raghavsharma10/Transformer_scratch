def get_dev_interface(devid):
    """
    Function takes devid as input to RESTFUL call to HP IMC platform
    :param devid: requires devid as the only input
    :return: list object which contains a dictionary per interface
    """
    # checks to see if the imc credentials are already available
    if auth is None or url is None:
        set_imc_creds()
    global r
    get_dev_interface_url = "/imcrs/plat/res/device/" + str(devid) + \
                            "/interface?start=0&size=1000&desc=false&total=false"
    f_url = url + get_dev_interface_url
    payload = None
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=headers)
    # r.status_code
    if r.status_code == 200:
        int_list = (json.loads(r.text))['interface']
        return int_list
    else:
        print("An Error has occured")