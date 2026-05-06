def save(self, *args, **kwargs):
        """
        Override save() method to make sure that standard_name and
        systematic_name won't be null or empty, or consist of only space
        characters (such as space, tab, new line, etc).
        """
        empty_std_name = False
        if not self.standard_name or self.standard_name.isspace():
            empty_std_name = True

        empty_sys_name = False
        if not self.systematic_name or self.systematic_name.isspace():
            empty_sys_name = True

        if empty_std_name and empty_sys_name:
            raise ValueError(
                "Both standard_name and systematic_name are empty")

        super(Gene, self).save(*args, **kwargs)