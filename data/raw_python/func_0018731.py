def set_text(self, text):
        """Sets properties and text given a text field"""
        self.text = text
        try:
            self.properties = text_to_dict(text)
        except:
            traceback.print_exc()
            self.properties = None