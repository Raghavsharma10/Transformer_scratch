def cmd_ssh(options):
    """Connect to the specified instance via ssh.

    Finds instances that match the user specified args that are also
    in the 'running' state.  The target instance is determined, the
    required connection information is retreived (IP, key and ssh
    user-name), then an 'ssh' connection is made to the instance.

    Args:
        options (object): contains args and data from parser

    """
    import os
    import subprocess
    from os.path import expanduser
    options.inst_state = "running"
    (i_info, param_str) = gather_data(options)
    (tar_inst, tar_idx) = determine_inst(i_info, param_str, options.command)
    home_dir = expanduser("~")
    if options.user is None:
        tar_aminame = awsc.get_one_aminame(i_info[tar_idx]['ami'])
        options.user = cmd_ssh_user(tar_aminame,
                                    i_info[tar_idx]['tag']['Name'])
    else:
        debg.dprint("LoginUser set by user: ", options.user)
    os_spec = {"nt": ["powershell plink", "\\", "ppk"]}
    c_itm = os_spec.get(os.name, ["ssh", "/", "pem"])
    cmd_ssh_run = c_itm[0]
    if not options.nopem:
        cmd_ssh_run += (" -i {0}{1}.aws{1}{2}.{3}".
                        format(home_dir, c_itm[1], i_info[tar_idx]['ssh_key'],
                               c_itm[2]))
    else:
        debg.dprint("Connect string: ", "ssh {}@{}".
                    format(options.user, i_info[tar_idx]['pub_dns_name']))
    cmd_ssh_run += " {0}@{1}".format(options.user,
                                     i_info[tar_idx]['pub_dns_name'])
    print(cmd_ssh_run)
    subprocess.call(cmd_ssh_run, shell=True)