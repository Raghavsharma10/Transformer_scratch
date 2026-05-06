def set_inteface_up(devid, ifindex, auth, url):
    """
    function takest devid and ifindex of specific device and interface and issues a RESTFUL call to "undo shut" the spec
    ified interface on the target device.
    :param devid: int or str value of the target device
    :param ifindex: int or str value of the target interface
    :return: HTTP status code 204 with no values.
    """
    set_int_up_url = "/imcrs/plat/res/device/" + str(devid) + "/interface/" + str(ifindex) + "/up"
    f_url = url + set_int_up_url
    try:
        r = requests.put(f_url, auth=auth,
                     headers=HEADERS)  # creates the URL using the payload variable as the contents
        if r.status_code == 204:
            return r.status_code
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " set_inteface_up: An Error has occured"