def run(self, plugin_manager=None):
        """Run the haas test runner.

        This will load and configure the selected plugins, set up the
        environment and begin test discovery, loading and running.

        Parameters
        ----------
        plugin_manager : haas.plugin_manager.PluginManager
            [Optional] Override the use of the default plugin manager.

        """
        if plugin_manager is None:
            plugin_manager = PluginManager()
        plugin_manager.add_plugin_arguments(self.parser)

        args = self.parser.parse_args(self.argv[1:])

        environment_plugins = plugin_manager.get_enabled_hook_plugins(
            plugin_manager.ENVIRONMENT_HOOK, args)
        runner = plugin_manager.get_driver(
            plugin_manager.TEST_RUNNER, args)

        with PluginContext(environment_plugins):
            loader = Loader()
            discoverer = plugin_manager.get_driver(
                plugin_manager.TEST_DISCOVERY, args, loader=loader)
            suites = [
                discoverer.discover(
                    start=start,
                    top_level_directory=args.top_level_directory,
                    pattern=args.pattern,
                )
                for start in args.start
            ]
            if len(suites) == 1:
                suite = suites[0]
            else:
                suite = loader.create_suite(suites)
            test_count = suite.countTestCases()
            result_handlers = plugin_manager.get_enabled_hook_plugins(
                plugin_manager.RESULT_HANDLERS, args, test_count=test_count)

            result_collector = ResultCollector(
                buffer=args.buffer, failfast=args.failfast)

            for result_handler in result_handlers:
                result_collector.add_result_handler(result_handler)

            result = runner.run(result_collector, suite)
            return not result.wasSuccessful()