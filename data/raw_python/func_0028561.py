def get_custom_views(auth, url,name=None,headers=HEADERS):
    """
    function requires no input and returns a list of dictionaries of custom views from an HPE IMC. Optional name
    argument will return only the specified view.
    :param name: str containing the name of the desired custom view
    :return: list of dictionaties containing attributes of the custom views
    """
    if name is None:
        get_custom_view_url = '/imcrs/plat/res/view/custom?resPrivilegeFilter=false&desc=false&total=false'
    elif name is not None:
        get_custom_view_url = '/imcrs/plat/res/view/custom?resPrivilegeFilter=false&name='+name+'&desc=false&total=false'
    f_url = url + get_custom_view_url
    r = requests.get(f_url, auth=auth, headers=headers)
    try:
        if r.status_code == 200:
            custom_view_list = (json.loads(r.text))["customView"]
            if type(custom_view_list) == dict:
                custom_view_list = [custom_view_list]
                return custom_view_list
            else:
                return custom_view_list
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + ' get_custom_views: An Error has occured'