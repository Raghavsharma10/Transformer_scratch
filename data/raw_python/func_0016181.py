def on_entry_click(self, event):
		"""
		function that gets called whenever entry is clicked
		"""
		if event.widget.config('fg') [4] == 'grey':
		   event.widget.delete(0, "end" ) # delete all the text in the entry
		   event.widget.insert(0, '') #Insert blank for user input
		   event.widget.config(fg = 'black')