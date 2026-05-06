def _set_configspec(self, section, copy):
        """
        Called by validate. Handles setting the configspec on subsections
        including sections to be validated by __many__
        """
        configspec = section.configspec
        many = configspec.get('__many__')
        if isinstance(many, dict):
            for entry in section.sections:
                if entry not in configspec:
                    section[entry].configspec = many

        for entry in configspec.sections:
            if entry == '__many__':
                continue
            if entry not in section:
                section[entry] = {}
                section[entry]._created = True
                if copy:
                    # copy comments
                    section.comments[entry] = configspec.comments.get(entry, [])
                    section.inline_comments[entry] = configspec.inline_comments.get(entry, '')

            # Could be a scalar when we expect a section
            if isinstance(section[entry], Section):
                section[entry].configspec = configspec[entry]