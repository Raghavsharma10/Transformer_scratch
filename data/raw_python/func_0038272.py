def _add_generate_sub_commands(self):
        """
        Sub commands for generating models for usage by clients.
        Currently supports Google Closure.
        """

        gen_parser = self._subparsers_handle.add_parser(
            name="gen",
            help="generate client side model stubs, filters"
            )

        gen_parser.add_argument(
            "-t",
            "--template",
            choices=['closure.model', 'closure.filter'],
            default='closure.model',
            required=True,
            dest="template",
            help="template to use for client side code generation"
            )

        gen_parser.add_argument(
            "-m",
            "--model",
            required=True,
            dest="models_definition",
            help="path to models definition file or package"
            )

        gen_parser.add_argument(
            "-o",
            "--output",
            default=".",
            dest="output",
            help="output path for generated code"
            )

        gen_parser.add_argument(
            "-n",
            "--namespace",
            required=True,
            dest="namespace",
            help="namespace to use with template e.g prestans.data.model"
            )

        gen_parser.add_argument(
            "-fn",
            "--filter-namespace",
            required=False,
            default=None,
            dest="filter_namespace",
            help="filter namespace to use with template e.g prestans.data.filter"
            )