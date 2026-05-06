def from_parse_args(cls, args):
        """Constructor from command line args.

        :param args: parse command line arguments
        :type args: argparse.ArgumentParser

        """

        return cls(args.migration_file,
                   args.database,
                   db_user=args.db_user,
                   db_password=args.db_password,
                   db_port=args.db_port,
                   db_host=args.db_host,
                   mode=args.mode,
                   allow_serie=args.allow_serie,
                   force_version=args.force_version,
                   web_host=args.web_host,
                   web_port=args.web_port,
                   web_custom_html=args.web_custom_html,
                   )