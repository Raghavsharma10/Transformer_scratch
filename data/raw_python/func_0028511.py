def delete_plat_operator(operator,auth, url, headers=HEADERS):
    """
    Function to set the password of an existing operator
    :param operator: str Name of the operator account
    :param password: str New password
    :param url: str url of IMC server, see requests library docs for more info
    :param auth: str see requests library docs for more info
    :param headers: json formated string. default values set in module
    :return:
    """
    #oper_id = None
    plat_oper_list = get_plat_operator(auth, url)
    for i in plat_oper_list:
        if operator == i['name']:
            oper_id = i['id']
    if oper_id == None:
        return("\n User does not exist")
    delete_plat_operator_url = "/imcrs/plat/operator/"
    f_url = url + delete_plat_operator_url + str(oper_id)
    r = requests.delete(f_url, auth=auth, headers=headers)
    try:
        if r.status_code == 204:
            print("\n Operator: " + operator +
                  " was successfully deleted")
            return r.status_code
    except requests.exceptions.RequestException as e:
        print ("Error:\n" + str(e) + ' delete_plat_operator: An Error has occured')
        return "Error:\n" + str(e) + ' delete_plat_operator: An Error has occured'