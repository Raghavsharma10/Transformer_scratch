def select2(self, box, drop, text):
        '''
        :param box: the locator for Selection Box
        :param drop: the locator for Selection Dropdown
        :param text: the text value to select or the index of the option to select
        :return: True
        :example: https://github.com/ldiary/marigoso/blob/master/notebooks/handling_select2_controls_in_selenium_webdriver.ipynb
        '''
        if not self.is_available(drop):
            if isinstance(box, str):
                self.get_element(box).click()
            else:
                box.click()
        ul_dropdown = self.get_element(drop)
        options = ul_dropdown.get_children('tag=li')
        if isinstance(text, int):
            options[text].click()
            return True

        for option in options:
            if option.text == text:
                option.click()
                return True
        print("[Error!] Selection not found: {}".format(text))
        print("Available Selections\n {}".format([option.text for option in options]))