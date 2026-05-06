def run(self):
        """The run loop. Returns self.destroy()"""
        while self.running:
            self.update()
            self.render()
            self.update_screen()

        return self.destroy()