def on_focusout(self, event, a):
		"""
		function that gets called whenever anywhere except entry is clicked
		"""
		if event.widget.get() == '':
			event.widget.insert(0, default_text[a])
			event.widget.config(fg = 'grey')