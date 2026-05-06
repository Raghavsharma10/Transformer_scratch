def delete_cfg_template(template_name, auth, url):
    """Uses the get_template_id() funct to gather the template_id
    to craft a url which is sent to the IMC server using a Delete Method
    :param template_name: str containing the entire contents of the configuration segment

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: If successful, return int of status.code 204.

    :rtype: Int

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.icc import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> delete_cfg_template('CW7SNMP.cfg', auth.creds, auth.url)


    """
    file_id = get_template_id(template_name, auth, url)
    f_url = url + "/imcrs/icc/confFile/" + str(file_id)
    response = requests.delete(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 204:
            print("Template successfully Deleted")
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " delete_cfg_template: An Error has occured"