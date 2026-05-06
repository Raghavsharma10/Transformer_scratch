def _update_python_paths(self):
        """ Append the workflow and libraries paths to the PYTHONPATH. """
        for path in self._config['workflows'] + self._config['libraries']:
            if os.path.isdir(os.path.abspath(path)):
                if path not in sys.path:
                    sys.path.append(path)
            else:
                raise ConfigLoadError(
                    'Workflow directory {} does not exist'.format(path))