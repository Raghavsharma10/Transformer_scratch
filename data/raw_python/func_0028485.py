def get_real_time_locate(ipAddress):
    """
    function takes the ipAddress of a specific host and issues a RESTFUL call to get the device and interface that the
    target host is currently connected to.
    :param ipAddress: str value valid IPv4 IP address
    :return: dictionary containing hostIp, devId, deviceIP, ifDesc, ifIndex
    """
    if auth is None or url is None:  # checks to see if the imc credentials are already available
        set_imc_creds()
    real_time_locate_url = "/imcrs/res/access/realtimeLocate?type=2&value=" + str(ipAddress) + "&total=false"
    f_url = url + real_time_locate_url
    r = requests.get(f_url, auth=auth, headers=headers)  # creates the URL using the payload variable as the contents
    if r.status_code == 200:
        return json.loads(r.text)['realtimeLocation']

    else:
        print(r.status_code)
        print("An Error has occured")