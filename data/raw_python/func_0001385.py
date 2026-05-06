def from_yaml(cls, file_path=None):
        """Create collection from a YAML file."""
        try:
            import yaml
        except ImportError:  # pragma: no cover
            yaml = None
        if not yaml:
            import sys
            sys.exit('PyYAML is not installed, but is required in order to parse YAML files.'
                     '\nTo install, run:\n$ pip install PyYAML\nor visit'
                     ' http://pyyaml.org/wiki/PyYAML for instructions.')

        with io.open(file_path, encoding=text_type('utf-8')) as stream:
            users_yaml = yaml.safe_load(stream)
            if isinstance(users_yaml, dict):
                return cls.construct_user_list(raw_users=users_yaml.get('users'))
            else:
                raise ValueError('No YAML object could be decoded')