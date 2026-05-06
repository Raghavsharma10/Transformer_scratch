def create_custom_views(auth, url,name=None, upperview=None):
    """
    function takes no input and issues a RESTFUL call to get a list of custom views from HPE IMC. Optioanl Name input
    will return only the specified view.
    :param name: string containg the name of the desired custom view
    :return: list of dictionaries containing attributes of the custom views.
    """
    create_custom_views_url = '/imcrs/plat/res/view/custom?resPrivilegeFilter=false&desc=false&total=false'
    f_url = url + create_custom_views_url
    if upperview is None:
        payload = '''{ "name": "''' + name + '''",
         "upLevelSymbolId" : ""}'''
        print (payload)
    else:
        parentviewid = get_custom_views(auth, url, upperview)[0]['symbolId']
        payload = '''{ "name": "'''+name+ '''",
        "upLevelSymbolId" : "'''+str(parentviewid)+'''"}'''
        print (payload)
    r = requests.post(f_url, data = payload, auth=auth, headers=HEADERS)  # creates the URL using the payload variable as the contents
    try:
        if r.status_code == 201:
            return 'View ' + name +' created successfully'
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + ' get_custom_views: An Error has occured'