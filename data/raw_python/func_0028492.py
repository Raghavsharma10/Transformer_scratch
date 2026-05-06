def set_inteface_up(devid, ifindex):
    """
    function takest devid and ifindex of specific device and interface and issues a RESTFUL call to "undo shut" the spec
    ified interface on the target device.
    :param devid: int or str value of the target device
    :param ifindex: int or str value of the target interface
    :return: HTTP status code 204 with no values.
    """
    if auth is None or url is None:  # checks to see if the imc credentials are already available
        set_imc_creds()
    set_int_up_url = "/imcrs/plat/res/device/" + str(devid) + "/interface/" + str(ifindex) + "/up"
    f_url = url + set_int_up_url
    payload = None
    r = requests.put(f_url, auth=auth,
                     headers=headers)  # creates the URL using the payload variable as the contents
    print(r.status_code)
    if r.status_code == 204:
        return r.status_code
    else:
        print("An Error has occured")