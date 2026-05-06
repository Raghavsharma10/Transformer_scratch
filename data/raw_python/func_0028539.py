def set_operator_password(operator, password, auth, url):
    """
    Function to set the password of an existing operator

    :param operator: str Name of the operator account

    :param password: str New password

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: int of 204 if successfull,

    :rtype: int

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.operator import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> operator = { "fullName" : "test administrator", "sessionTimeout" : "30",
                     "password" :  "password","operatorGroupId" : "1",
                     "name" : "testadmin","desc" : "test admin account",
                     "defaultAcl" : "","authType"  : "0"}

    >>> new_operator = create_operator(operator, auth.creds, auth.url)

    >>> set_new_password = set_operator_password('testadmin', 'newpassword', auth.creds, auth.url)

    >>> assert type(set_new_password) is int

    >>> assert set_new_password == 204

       """
    if operator is None:
        operator = input(
            '''\n What is the username you wish to change the password?''')
    oper_id = ''
    authtype = None
    plat_oper_list = get_plat_operator(auth, url)
    for i in plat_oper_list:
        if i['name'] == operator:
            oper_id = i['id']
            authtype = i['authType']
    if oper_id == '':
        return "User does not exist"
    change_pw_url = "/imcrs/plat/operator/"
    f_url = url + change_pw_url + oper_id
    if password is None:
        password = input(
            '''\n ============ Please input the operators new password:\n ============  ''')
    payload = json.dumps({'password': password, 'authType': authtype})
    response = requests.put(f_url, data=payload, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 204:
            # print("Operator:" + operator +
            # " password was successfully changed")
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' set_operator_password: An Error has occured'