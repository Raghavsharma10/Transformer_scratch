def get_trap_definitions():
    """Takes in no param as input to fetch SNMP TRAP definitions from HP IMC RESTFUL API
    :param None
    :return: object of type list containing the device asset details
    """
    # checks to see if the imc credentials are already available
    if auth is None or url is None:
        set_imc_creds()
    global r
    get_trap_def_url = "/imcrs/fault/trapDefine/sync/query?enterpriseId=1.3.6.1.4.1.11&size=10000"
    f_url = url + get_trap_def_url
    payload = None
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=headers)
    # r.status_code
    if r.status_code == 200:
        trap_def_list = (json.loads(r.text))
        return trap_def_list['trapDefine']
    else:
        print("get_dev_asset_details:  An Error has occured")