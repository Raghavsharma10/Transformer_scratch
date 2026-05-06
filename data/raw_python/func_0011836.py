def _find_all_first_files(self, item):
        """
        Does not support the full range of ways rar can split
        as it'd require reading the file to ensure you are using the
        correct way.
        """
        for listed_item in item.list():
            new_style = re.findall(r'(?i)\.part(\d+)\.rar^', listed_item.id)
            if new_style:
                if int(new_style[0]) == 1:
                    yield 'new', listed_item
            elif listed_item.id.lower().endswith('.rar'):
                yield 'old', listed_item