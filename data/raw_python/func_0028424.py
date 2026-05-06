def create_operator(operator, auth, url,headers=HEADERS):
    """
    Function takes input of dictionary operator with the following keys
    operator = { "fullName" : ""   ,
             "sessionTimeout" : "",
             "password" :  "",
             "operatorGroupId" : "",
             "name" : "",
             "desc" : "",
             "defaultAcl" : "",
             "authType"  : ""}
    converts to json and issues a HTTP POST request to the HPE IMC Restful API

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param operator: dictionary with the required operator key-value pairs as defined above.

    :param headers: json formated string. default values set in module

    :return:

    :rtype:


    >>> import json

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.operator import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> operator = '''{ "fullName" : "test administrator", "sessionTimeout" : "30","password" :  "password","operatorGroupId" : "1","name" : "testadmin","desc" : "test admin account","defaultAcl" : "","authType"  : "0"}'''

    >>> operator = json.loads(operator)

    >>> delete_if_exists = delete_plat_operator('testadmin', auth.creds, auth.url)

    >>> new_operator = create_operator(operator, auth.creds, auth.url)

    >>> assert type(new_operator) is int

    >>> assert new_operator == 201

    >>> fail_operator_create = create_operator(operator, auth.creds, auth.url)

    >>> assert type(fail_operator_create) is int

    >>> assert fail_operator_create == 409

    """
    create_operator_url = '/imcrs/plat/operator'
    f_url = url + create_operator_url
    payload = json.dumps(operator, indent=4)
    # creates the URL using the payload variable as the contents
    r = requests.post(f_url, data=payload, auth=auth, headers=headers)
    try:
        if r.status_code == 409:
            #print("Operator Already Exists")
            return r.status_code
        elif r.status_code == 201:
            return r.status_code
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + ' create_operator: An Error has occured'