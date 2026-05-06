def run(self):
        """
        pop up a dialog box and return when the user has closed it
        """
        response = None
        root = tkinter.Tk()
        root.withdraw()
        while response is not True:
            response = tkinter.messagebox.askokcancel(title=self.title, message=self.pre_message)
        if self.post_message:
            print(self.post_message)
        self.exit_time = time.time()