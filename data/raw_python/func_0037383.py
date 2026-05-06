def _normalize_tags(self, tags):
        '''
        Coerces tags to lowercase strings

        Parameters
        ----------
        tags: list or tuple of strings

        '''

        lowered_str_tags = []
        for tag in tags:
            lowered_str_tags.append(str(tag).lower())

        return lowered_str_tags