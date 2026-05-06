def handle_select(self):
        """Handle user's input in list mode."""
        self.selected = input('>> ')
        if self.selected in ['Q', 'q']:
            sys.exit(1)
        elif self.selected in ['B', 'b']:
            self.back_to_menu = True
            return True
        elif is_num(self.selected):
            if 0 <= int(self.selected) <= len(self.hrefs) - 1:
                self.back_to_menu = False
                return True
            else:
                print(Colors.FAIL +
                      'Wrong index. ' +
                      'Please select an appropiate one or other option.' +
                      Colors.ENDC)
                return False
        else:
            print(Colors.FAIL +
                  'Invalid input. ' +
                  'Please select an appropiate one or other option.' +
                  Colors.ENDC)
            return False