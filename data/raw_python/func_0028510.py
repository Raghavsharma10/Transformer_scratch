def get_plat_operator(auth, url,headers=HEADERS):
    '''
    Funtion takes no inputs and returns a list of dictionaties of all of the operators currently configured on the HPE
    IMC system
    :return: list of dictionaries
    '''
    get_operator_url = '/imcrs/plat/operator?start=0&size=1000&orderBy=id&desc=false&total=false'
    f_url = url + get_operator_url
    try:
        r = requests.get(f_url, auth=auth, headers=headers)
        plat_oper_list = json.loads(r.text)
        return plat_oper_list['operator']
    except requests.exceptions.RequestException as e:
        print ("Error:\n" + str(e) + ' get_plat_operator: An Error has occured')
        return "Error:\n" + str(e) + ' get_plat_operator: An Error has occured'