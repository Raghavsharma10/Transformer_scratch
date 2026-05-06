def merge_extends(self, target, extends, inherit_key="inherit", inherit=False):
        """Merge extended dicts
        """
        if isinstance(target, dict):
            if inherit and inherit_key in target and not to_boolean(target[inherit_key]):
                return
            if not isinstance(extends, dict):
                raise ValueError("Unable to merge: Dictionnary expected")
            for key in extends:
                if key not in target:
                    target[str(key)] = extends[key]
                else:
                    self.merge_extends(target[key], extends[key], inherit_key, True)
        elif isinstance(target, list):
            if not isinstance(extends, list):
                raise ValueError("Unable to merge: List expected")
            target += extends