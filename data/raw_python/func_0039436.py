def show_help(self):
        """ Redraw Help screen and wait for any input to leave """
        self.s.move(1,0)
        self.s.clrtobot()
        self.set_header('Help'.center(self.pad_w))
        self.set_footer(' ESC or \'q\' to return to main menu')
        self.s.refresh()
        self.current_pad = 'help'
        self.refresh_current_pad()