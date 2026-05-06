def cmd_ssh_user(tar_aminame, inst_name):
    """Calculate instance login-username based on image-name.

    Args:
        tar_aminame (str): name of the image instance created with.
        inst_name (str): name of the instance.
    Returns:
        username (str): name for ssh based on AMI-name.

    """
    if tar_aminame == "Unknown":
        tar_aminame = inst_name
    # first 5 chars of AMI-name can be anywhere in AMI-Name
    userlu = {"ubunt": "ubuntu", "debia": "admin", "fedor": "root",
              "cento": "centos", "openb": "root"}
    usertemp = ['name'] + [value for key, value in list(userlu.items())
                           if key in tar_aminame.lower()]
    usertemp = dict(zip(usertemp[::2], usertemp[1::2]))
    username = usertemp.get('name', 'ec2-user')
    debg.dprint("loginuser Calculated: ", username)
    return username