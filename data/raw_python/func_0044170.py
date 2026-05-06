def append(self, newconfig):
        """ Append another site config to current instance.

        All ``newconfig`` attributes are appended one by one to ours.
        Order matters, eg. current instance values will come first when
        merging.

        Thus, if you plan to use some sort of global site config with
        more generic directives, append it last for specific directives
        to be tried first.

        .. note:: this method is also aliased to :meth:`merge`.
        """

        # Check for commands where we accept multiple statements (no test_url)
        for attr_name in (
            'title', 'body', 'author', 'date',
            # `language` is fixed in reset() and
            # not supported in siteconfig syntax.
            'strip', 'strip_id_or_class', 'strip_image_src',
            'single_page_link', 'single_page_link_in_feed',
            'next_page_link', 'http_header'
        ):
            # Append to ordered set. We keep ordering, but no duplicates.
            current_set = getattr(self, attr_name)
            for val in getattr(newconfig, attr_name):
                # Too bad ordered set has no .union() method.
                current_set.add(val)
            setattr(self, attr_name, current_set)

        # Check for single statement commands;
        # we do not overwrite existing values.
        for attr_name in (
            'parser', 'tidy', 'prune', 'autodetect_on_failure'
        ):
            if getattr(self, attr_name) is None:
                if getattr(newconfig, attr_name) is None:
                    setattr(self, attr_name, self.defaults[attr_name])
                else:
                    setattr(self, attr_name, getattr(newconfig, attr_name))

        # HEADS UP: PHP → Python port.
        if self.parser == 'libxml':
            self.parser = 'lxml'

        for attr_name in ('find_string', 'replace_string', ):
            # Find/replace strings are lists, we extend.
            getattr(self, attr_name).extend(getattr(newconfig, attr_name))

        if self.find_string:
            # This will ease replacements in the extractor.
            self.replace_patterns = zip(
                self.find_string, self.replace_string)

        else:
            self.replace_patterns = None