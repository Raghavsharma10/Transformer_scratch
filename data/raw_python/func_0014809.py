def parse_cli_args(self):
        ''' Command line argument processing '''

        parser = argparse.ArgumentParser(
                description='Produce an Ansible Inventory file based on GCE')
        parser.add_argument('--list', action='store_true', default=True,
                           help='List instances (default: True)')
        parser.add_argument('--host', action='store',
                           help='Get all information about an instance')
        parser.add_argument('--pretty', action='store_true', default=False,
                           help='Pretty format (default: False)')
        parser.add_argument(
            '--refresh-cache', action='store_true', default=False,
            help='Force refresh of cache by making API requests (default: False - use cache files)')
        self.args = parser.parse_args()