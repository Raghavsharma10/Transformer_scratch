def create_cmd(self, args):
        '''
        'create' sub-command
        :param args: cli arguments
        :return:
        '''
        cmd = args.get('cmd_create')
        if cmd == 'conf':
            conf_file = args['conf_file']
            conf_id = args['id']
            return self.load_xml_conf(conf_file, conf_id)
        else:
            print("Error: Create %s is invalid or not implemented" % cmd)