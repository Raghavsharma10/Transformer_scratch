def get_args():
    """
    Supports the command-line arguments listed below.
    """

    parser = argparse.ArgumentParser(
        description="Provision multiple VM's through mist.io. You can get "
        "information returned with the name of the virtual machine created "
        "and its main mac and ip address in IPv4 format. A post-script can be "
        "specified for post-processing.")
    parser.add_argument('-b', '--basename', nargs=1, required=True,
                        help='Basename of the newly deployed VMs',
                        dest='basename', type=str)
    parser.add_argument('-d', '--debug', required=False,
                        help='Enable debug output', dest='debug',
                        action='store_true')
    parser.add_argument('-i', '--print-ips', required=False,
                        help='Enable IP output', dest='ips',
                        action='store_true')
    parser.add_argument('-m', '--print-macs', required=False,
                        help='Enable MAC output', dest='macs',
                        action='store_true')
    parser.add_argument('-l', '--log-file', nargs=1, required=False,
                        help='File to log to (default = stdout)',
                        dest='logfile', type=str)
    parser.add_argument('-n', '--number', nargs=1, required=False,
                        help='Amount of VMs to deploy (default = 1)',
                        dest='quantity', type=int, default=[1])
    parser.add_argument('-M', '--monitoring', required=False,
                        help='Enable monitoring on the virtual machines',
                        dest='monitoring', action='store_true')
    parser.add_argument('-B', '--backend-name', required=False,
                        help='The name of the backend to use for provisioning.'
                        ' Defaults to the first available backend',
                        dest='backend_name', type=str)
    parser.add_argument('-I', '--image-id', required=True,
                        help='The image to deploy', dest='image_id')
    parser.add_argument('-S', '--size-id', required=True,
                        help='The id of the size/flavor to use',
                        dest='size_id')
    parser.add_argument('-N', '--networks', required=False, nargs='+',
                        help='The ids of the networks to assign to the VMs',
                        dest='networks')
    parser.add_argument('-s', '--post-script', nargs=1, required=False,
                        help='Script to be called after each VM is created and'
                        ' booted.', dest='post_script', type=str)
    parser.add_argument('-P', '--script-params', nargs=1, required=False,
                        help='Script to be called after each VM is created and'
                        ' booted.', dest='script_params', type=str)
    parser.add_argument('-H', '--host', required=False,
                        help='mist.io instance to connect to', dest='host',
                        type=str, default='https://mist.io')
    parser.add_argument('-u', '--user', nargs=1, required=False,
                        help='email registered to mist.io', dest='username',
                        type=str)
    parser.add_argument('-p', '--password', nargs=1, required=False,
                        help='The password with which to connect to the host. '
                        'If not specified, the user is prompted at runtime for'
                        ' a password', dest='password', type=str)
    parser.add_argument('-v', '--verbose', required=False,
                        help='Enable verbose output', dest='verbose',
                        action='store_true')
    parser.add_argument('-w', '--wait-max', nargs=1, required=False,
                        help='Maximum amount of seconds to wait when gathering'
                        ' information (default = 600)', dest='maxwait',
                        type=int, default=[600])
    parser.add_argument('-f', '--associate-floating-ip', required=False, action='store_true',
                        help='Auto-associates floating ips to vms in Openstack backens',
                        dest='associate_floating_ip',)
    args = parser.parse_args()
    return args