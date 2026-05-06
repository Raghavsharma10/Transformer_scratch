def main():
    '''
    Simple examples
    '''
    args = parse_arguments()
    if args.askpass:
        password = getpass.getpass("Password: ")
    else:
        password = None

    if args.asksudopass:
        sudo = True
        sudo_pass = getpass.getpass("Sudo password[default ssh password]: ")
        if len(sudo_pass) == 0:
            sudo_pass = password
        sudo_user = 'root'
    else:
        sudo = False
        sudo_pass = None
        sudo_user = None

    if not args.username:
        username = getpass.getuser()
    else:
        username = args.username

    host_list = args.hosts
    os.environ["ANSIBLE_HOST_KEY_CHECKING"] = "False"

    execute_ping(host_list, username, password,
                 sudo=sudo, sudo_user=sudo_user, sudo_pass=sudo_pass)