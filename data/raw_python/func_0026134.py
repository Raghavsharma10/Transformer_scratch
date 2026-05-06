def toggle(self, event=None):
        """Toggle value between Yes and No"""
        if self.choice.get() == "yes":
            self.rbno.select()
        else:
            self.rbyes.select()
        self.widgetEdited()