def create_custom_views(name=None, upperview=None):
    """
    function takes no input and issues a RESTFUL call to get a list of custom views from HPE IMC. Optioanl Name input
    will return only the specified view.
    :param name: string containg the name of the desired custom view
    :return: list of dictionaries containing attributes of the custom views.
    """
    if auth is None or url is None:  # checks to see if the imc credentials are already available
        set_imc_creds()
    create_custom_views_url = '/imcrs/plat/res/view/custom?resPrivilegeFilter=false&desc=false&total=falsee'
    f_url = url + create_custom_views_url
    if upperview is None:
        payload = '''{ "name": "''' + name + '''",
         "upLevelSymbolId" : ""}'''
    else:
        parentviewid = get_custom_views(upperview)[0]['symbolId']
        payload = '''{
                         "name": "'''+name+ '''"upperview" : "'''+str(parentviewid)+'''"}'''
        print (payload)
    r = requests.post(f_url, data = payload, auth=auth, headers=headers)  # creates the URL using the payload variable as the contents
    if r.status_code == 201:
        return 'View ' + name +' created successfully'
    else:
        print(r.status_code)
        print("An Error has occured")