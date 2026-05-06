def special_format_field(self, obj, format_spec):
        """Know about any special formats"""
        if format_spec == "env":
            return "${{{0}}}".format(obj)
        elif format_spec == "from_env":
            if obj not in os.environ:
                raise NoSuchEnvironmentVariable(wanted=obj)
            return os.environ[obj]