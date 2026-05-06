def get_dev_asset_details(ipaddress):
    """Takes in ipaddress as input to fetch device assett details from HP IMC RESTFUL API
    :param ipaddress: IP address of the device you wish to gather the asset details
    :return: object of type list containing the device asset details
    """
    # checks to see if the imc credentials are already available
    if auth is None or url is None:
        set_imc_creds()
    global r
    get_dev_asset_url = "/imcrs/netasset/asset?assetDevice.ip=" + str(ipaddress)
    f_url = url + get_dev_asset_url
    payload = None
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=headers)
    # r.status_code
    if r.status_code == 200:
        dev_asset_info = (json.loads(r.text))
        if len(dev_asset_info) > 0:
            dev_asset_info = dev_asset_info['netAsset']
        if type(dev_asset_info) == dict:
            dev_asset_info = [dev_asset_info]
        if type(dev_asset_info) == list:
            dev_asset_info[:] = [dev for dev in dev_asset_info if dev.get('deviceIp') == ipaddress]
        return dev_asset_info
    else:
        print("get_dev_asset_details:  An Error has occured")