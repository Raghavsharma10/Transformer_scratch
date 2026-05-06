def validate_format(self, **kwargs):
        """Call ConfigParser to validate config

        Args:
            kwargs: are passed to :class:`configparser.ConfigParser`
        """
        args = dict(
            dict_type=self._dict,
            allow_no_value=self._allow_no_value,
            inline_comment_prefixes=self._inline_comment_prefixes,
            strict=self._strict,
            empty_lines_in_values=self._empty_lines_in_values
        )
        args.update(kwargs)
        parser = ConfigParser(**args)
        updated_cfg = str(self)
        parser.read_string(updated_cfg)