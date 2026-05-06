def load_yaml_by_path(cls, path, log_debug=False):
        """Load a yaml file that is at given path,
        if the path is not a string, it is assumed it's a file-like object"""
        try:
            if isinstance(path, six.string_types):
                return yaml.load(open(path, 'r'), Loader=Loader) or {}
            else:
                return yaml.load(path, Loader=Loader) or {}
        except (yaml.scanner.ScannerError, yaml.parser.ParserError) as e:
            log_level = logging.DEBUG if log_debug else logging.WARNING
            logger.log(log_level, 'Yaml error in {path} (line {ln}, column {col}): {err}'.
                       format(path=path,
                              ln=e.problem_mark.line,
                              col=e.problem_mark.column,
                              err=e.problem))
            return None