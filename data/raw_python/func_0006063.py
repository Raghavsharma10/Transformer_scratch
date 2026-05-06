def checkbox_check(self, force_check=False):
        """
        Wrapper to check a  checkbox
        """
        if not self.get_attribute('checked'):
            self.click(force_click=force_check)