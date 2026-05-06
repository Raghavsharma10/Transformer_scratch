def on_click(self, event):
        """
        A function that should be overwritten by a plugin that wishes to react
        to events, if it wants to perform any action other than running the
        supplied command related to a button.

        event: A dictionary passed from i3bar (after being decoded from JSON)
        that has the folowing format:

        event = {'name': 'my_plugin', 'x': 231, 'y': 423}
        Note: It is also possible to have an instance key, but i3situation
        doesn't set it.
        """
        if event['button'] == 1 and 'button1' in self.options:
            subprocess.call(self.options['button1'].split())
        elif event['button'] == 2 and 'button2' in self.options:
            subprocess.call(self.options['button2'].split())
        elif event['button'] == 3 and 'button3' in self.options:
            subprocess.call(self.options['button3'].split())