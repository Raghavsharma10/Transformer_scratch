def confirm(prompt, default=None, show_default=True, abort=False, input_function=None):
	'''Prompts for confirmation from the user.
	'''
	valid = {
		'yes': True,
		'y': True,
		'no': False,
		'n': False
	}
	input_function = get_input_fn(input_function)
	if default not in ['yes', 'no', None]:
		default = None
	if show_default:
		prompt = '{} [{}/{}]: '.format(prompt,
				'Y' if default == 'yes' else 'y',
				'N' if default == 'no' else 'n')
	while True:
		choice = prompt_fn(input_function, prompt, default).lower()
		if choice in valid:
			if valid[choice] == False and abort:
				raise_abort()
			return valid[choice]
		else:
			echo('Please respond with "yes" or "no" (or "y" or "n").')