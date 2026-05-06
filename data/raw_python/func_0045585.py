def progressbar(
		iterable=None, length=None, label=None,
		show_eta=True, show_percent=None, show_pos=False, item_show_func=None,
		fill_char='#', empty_char='-', bar_template='%(label)s [%(bar)s] %(info)s', info_sep=' ',
		width=36, file=None, color=None):
	"""Create a progressbar that works in Jupyter/IPython notebooks and the terminal"""
	
	try:
		return IPyBackend(iterable, length, label=label,
			show_eta=show_eta, show_percent=show_percent, show_pos=show_pos,
			item_show_func=item_show_func, info_sep=info_sep)
	except (ImportError, RuntimeError): #fall back if ipython is not installed or no notebook is running
		return click.progressbar(
			iterable, length, label,
			show_eta, show_percent, show_pos, item_show_func,
			fill_char, empty_char, bar_template, info_sep,
			width, file, color)