def delete_plat_operator(operator, auth, url):
    """
    Function to set the password of an existing operator
    :param operator: str Name of the operator account

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: int of 204 if successfull

    :rtype: int

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.operator import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> success_delete_operator = delete_plat_operator('testadmin', auth.creds, auth.url)

    >>> assert type(success_delete_operator) is int

    >>> assert success_delete_operator == 204

    >>> fail_delete_operator = delete_plat_operator('testadmin', auth.creds, auth.url)

    >>> assert type(fail_delete_operator) is int

    >>> assert fail_delete_operator == 409

    """
    oper_id = None
    plat_oper_list = get_plat_operator(auth, url)
    for i in plat_oper_list:
        if operator == i['name']:
            oper_id = i['id']
        else:
            oper_id = None
    if oper_id is None:
        # print ("User does not exist")
        return 409
    f_url = url + "/imcrs/plat/operator/" + str(oper_id)
    response = requests.delete(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 204:
            # print("Operator: " + operator +
            #  " was successfully deleted")
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' delete_plat_operator: An Error has occured'