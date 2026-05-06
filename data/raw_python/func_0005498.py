def main():
    """Sample usage for this python module

    This main method simply illustrates sample usage for this python
    module.

    :return: None
    """
    log = logging.getLogger(mod_logger + '.main')
    parser = argparse.ArgumentParser(description='cons3rt deployment CLI')
    parser.add_argument('command', help='Command for the deployment CLI')
    parser.add_argument('--network', help='Name of the network')
    parser.add_argument('--name', help='Name of a deployment property to get')
    args = parser.parse_args()

    valid_commands = ['ip', 'device', 'prop']
    valid_commands_str = ','.join(valid_commands)

    # Get the command
    command = args.command.strip().lower()

    # Ensure the command is valid
    if command not in valid_commands:
        print('Invalid command found [{c}]\n'.format(c=command) + valid_commands_str)
        return 1

    if command == 'ip':
        if not args.network:
            print('Missed arg: --network, for the name of the network')
    elif command == 'device':
        if not args.network:
            print('Missed arg: --network, for the name of the network')
            return 1
        d = Deployment()
        print(d.get_device_for_network_linux(network_name=args.network))
    elif command == 'prop':
        if not args.name:
            print('Missed arg: --name, for the name of the property to retrieve')
            return 1
        d = Deployment()
        print(d.get_value(property_name=args.name))