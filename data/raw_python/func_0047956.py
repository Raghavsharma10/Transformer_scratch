def prompt(text, default=None, show_default=True, invisible=False,
           confirm=False, skip=False, type=None, input_function=None):
	'''Prompts for input from the user.
	'''
	t = determine_type(type, default)
	input_function = get_input_fn(input_function, invisible)
	if default is not None and show_default:
		text = '{} [{}]: '.format(text, default)
	while True:
		val = prompt_fn(input_function, text, default, t, skip, repeat=True)
		if not confirm or (skip and val is None):
			return val
		if val == prompt_fn(input_function, 'Confirm: ', default, t, repeat=True):
			return val
		echo('Error: The two values you entered do not match', True)