def delete_cfg_template(template_name, auth, url):
    '''Uses the get_template_id() funct to gather the template_id to craft a url which is sent to the IMC server using
    a Delete Method
    :param template_name: str containing the entire contents of the configuration segment

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: If successful, Boolean of type True

    :rtype: Boolean

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.icc import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> delete_cfg_template('CW7SNMP.cfg', auth.creds, auth.url)
    True

    >>> get_template_id('CW7SNMP.cfg', auth.creds, auth.url)
    'template not found'

    '''
    file_id = get_template_id(template_name, auth, url)
    delete_cfg_template_url = "/imcrs/icc/confFile/"+str(file_id)
    f_url = url + delete_cfg_template_url
    # creates the URL using the payload variable as the contents
    r = requests.delete(f_url, auth=auth, headers=HEADERS)
    #print (r.status_code)
    try:
        if r.status_code == 204:
            return True
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " delete_cfg_template: An Error has occured"