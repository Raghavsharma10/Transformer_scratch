def check_yamls(cls, dap):
        '''Check that all assistants and snippets are valid.

        Return list of DapProblems.'''
        problems = list()

        for yaml in dap.assistants_and_snippets:
            path = yaml + '.yaml'
            parsed_yaml = YamlLoader.load_yaml_by_path(dap._get_file(path, prepend=True))
            if parsed_yaml:
                try:
                    yaml_checker.check(path, parsed_yaml)
                except YamlError as e:
                    problems.append(DapProblem(exc_as_decoded_string(e), level=logging.ERROR))
            else:
                problems.append(DapProblem('Empty YAML ' + path, level=logging.WARNING))

        return problems