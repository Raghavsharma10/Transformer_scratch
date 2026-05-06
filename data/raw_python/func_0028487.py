def get_custom_views(name=None):
    """
    function takes no input and issues a RESTFUL call to get a list of custom views from HPE IMC. Optioanl Name input
    will return only the specified view.
    :param name: string containg the name of the desired custom view
    :return: list of dictionaries containing attributes of the custom views.
    """
    if auth is None or url is None:  # checks to see if the imc credentials are already available
        set_imc_creds()
    if name is None:
        get_custom_views_url = '/imcrs/plat/res/view/custom?resPrivilegeFilter=false&desc=false&total=false'
    elif name is not None:
        get_custom_views_url = '/imcrs/plat/res/view/custom?resPrivilegeFilter=false&name='+ name + '&desc=false&total=false'
    f_url = url + get_custom_views_url
    r = requests.get(f_url, auth=auth, headers=headers)  # creates the URL using the payload variable as the contents
    if r.status_code == 200:
        customviewlist = (json.loads(r.text))['customView']
        if type(customviewlist) is dict:
            customviewlist = [customviewlist]
            return customviewlist
        else:
            return customviewlist
    else:
        print(r.status_code)
        print("An Error has occured")