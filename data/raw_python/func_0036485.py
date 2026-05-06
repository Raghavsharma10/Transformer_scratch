def _extract_meta_value(self, tag):
        # type: (str, List[str]) -> str
        """Find a target value by `tag` from given meta data.

        :param tag: str
        :param meta_data: list
        :return: str
        """
        try:
            return [l[len(tag):] for l in self.meta_data if l.startswith(tag)][0]
        except IndexError:
            return '* Not Found *'