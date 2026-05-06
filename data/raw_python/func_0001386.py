def from_json(cls, file_path=None):
        """Create collection from a JSON file."""
        with io.open(file_path, encoding=text_type('utf-8')) as stream:
            try:
                users_json = json.load(stream)
            except ValueError:
                raise ValueError('No JSON object could be decoded')
            return cls.construct_user_list(raw_users=users_json.get('users'))