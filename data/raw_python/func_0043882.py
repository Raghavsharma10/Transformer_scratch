def login_as_bot():
	"""
	Login as the bot account "octogrid", if user isn't authenticated on Plotly
	"""

	plotly_credentials_file = join(
    	join(expanduser('~'), PLOTLY_DIRECTORY), PLOTLY_CREDENTIALS_FILENAME)

	if isfile(plotly_credentials_file):
		with open(plotly_credentials_file, 'r') as f:
			credentials = loads(f.read())

		if (credentials['username'] == '' or credentials['api_key'] == ''):
			plotly.sign_in(BOT_USERNAME, BOT_API_KEY)
	else:
		plotly.sign_in(BOT_USERNAME, BOT_API_KEY)