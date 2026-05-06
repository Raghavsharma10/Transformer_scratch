def add_subassistants_to(cls, parser, assistant_tuple, level, alias=None):
        """Adds assistant from given part of assistant tree and all its subassistants to
        a given argument parser.

        Args:
            parser: instance of devassistant_argparse.ArgumentParser
            assistant_tuple: part of assistant tree (see generate_argument_parser doc)
            level: level of subassistants that given assistant is at
        """
        name = alias or assistant_tuple[0].name
        p = parser.add_parser(name,
                              description=assistant_tuple[0].description,
                              argument_default=argparse.SUPPRESS)
        for arg in assistant_tuple[0].args:
            arg.add_argument_to(p)

        if len(assistant_tuple[1]) > 0:
            subparsers = cls._add_subparsers_required(p,
                dest=settings.SUBASSISTANT_N_STRING.format(level),
                title=cls.subparsers_str,
                description=cls.subparsers_desc)
            for subas_tuple in sorted(assistant_tuple[1], key=lambda x: x[0].name):
                cls.add_subassistants_to(subparsers, subas_tuple, level + 1)
        elif level == 1:
            subparsers = cls._add_subparsers_required(p,
                dest=settings.SUBASSISTANT_N_STRING.format(level),
                title=cls.subparsers_str,
                description=devassistant_argparse.ArgumentParser.no_assistants_msg)