def get_OS_UUID(cls, os):
        """
        Validate Storage OS and its UUID.

        If the OS is a custom OS UUID, don't validate against templates.
        """
        if os in cls.templates:
            return cls.templates[os]

        uuid_regexp = '^[0-9a-z]{8}-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{12}$'
        if re.search(uuid_regexp, os):
            return os

        raise Exception((
            "Invalid OS -- valid options are: 'CentOS 6.5', 'CentOS 7.0', "
            "'Debian 7.8', 'Debian 8.0' ,'Ubuntu 12.04', 'Ubuntu 14.04', 'Ubuntu 16.04', "
            "'Windows 2008', 'Windows 2012'"
        ))