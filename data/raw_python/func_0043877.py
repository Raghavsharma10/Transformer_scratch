def select_text(self, coordinate, text, select2drop=None):
        '''
        :param an element or a locator of the select element
        :param a selection text or selection index to be selected
        :param the select2 dropdown locator
        :return: True
        '''
        if not isinstance(coordinate, Select):
            if isinstance(coordinate, str):
                element = self.get_element(coordinate)
            else:
                element = coordinate
            if element.tag_name.lower() != "select":
                # Selenium's Select does not support non-select tags
                return self.select2(element, select2drop, text)
            selection = Select(element)
        else:
            selection = coordinate
            # TODO: Can't understand why Select made it private, replace Select in the future(?)
            element = coordinate._el
        if isinstance(text, int):
            if select2drop is not None:
                return self.select2(element, select2drop, text)
            return selection.select_by_index(text)
        try:
            selection.select_by_visible_text(text)
            return True
        except NoSuchElementException:
            available_selections = []
            for option in selection.options:
                if text in option.text:
                    selection.select_by_visible_text(option.text)
                    return True
                available_selections.append(option.text)
            print("[Error!] Selection not found: {}".format(text))
            print("Available Selections\n {}".format(available_selections))
        except ValueError:
            if select2drop is None:
                print("[Hint] You might be dealing with a select2 element. Try specifying select2drop locator.")
                raise
            # We are assuming we encountered a select2 selection box
            self.select2(element, select2drop, text)