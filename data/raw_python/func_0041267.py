def decode_file_args(self, argv: List[str]) -> List[str]:
        """
        Preprocess a configuration file.  The location of the configuration file is stored in the parser so that the
        FileOrURI action can add relative locations.
        :param argv: raw options list
        :return: options list with '--conf' references replaced with file contents
        """
        for i in range(0, len(argv) - 1):
            # TODO: take prefix into account
            if argv[i] == '--conf':
                del argv[i]
                conf_file = argv[i]
                del (argv[i])
                with open(conf_file) as config_file:
                    conf_args = shlex.split(config_file.read())
                    # We take advantage of a poential bug in the parser where you can say "foo -u 1 -u 2" and get
                    # 2 as a result
                    argv = self.fix_rel_paths(conf_args, conf_file) + argv
                return self.decode_file_args(argv)
        return argv