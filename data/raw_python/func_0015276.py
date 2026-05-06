def get_perm_prompt(cls, package_list):
        """
        Return text for prompt (do you want to install...), to install given packages.
        """
        if cls == PackageManager:
            raise NotImplementedError()
        ln = len(package_list)
        plural = 's' if ln > 1 else ''
        return cls.permission_prompt.format(num=ln, plural=plural)