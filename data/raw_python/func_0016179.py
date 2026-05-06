def main():
	"""
	main function
	"""
	s = ttk.Style()
	s.theme_use('clam')
	ents = makeform(root)
	root.mainloop()