def _publish_match(self, publish, names=False, name_only=False):
        """
        Check if publish name matches list of names or regex patterns
        """
        if names:
            for name in names:
                if not name_only and isinstance(name, re._pattern_type):
                    if re.match(name, publish.name):
                        return True
                else:
                    operand = name if name_only else [name, './%s' % name]
                    if publish in operand:
                        return True
            return False
        else:
            return True