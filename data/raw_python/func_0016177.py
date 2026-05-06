def task(ft):
	"""
	to create loading progress bar
	"""
	ft.pack(expand = True,  fill = BOTH,  side = TOP)
	pb_hD = ttk.Progressbar(ft, orient = 'horizontal', mode = 'indeterminate')
	pb_hD.pack(expand = True, fill = BOTH, side = TOP)
	pb_hD.start(50)
	ft.mainloop()