def checkbox_uncheck(self, force_check=False):
        """
        Wrapper to uncheck a checkbox
        """
        if self.get_attribute('checked'):
            self.click(force_click=force_check)