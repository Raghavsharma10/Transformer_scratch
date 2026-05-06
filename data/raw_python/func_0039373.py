def getUserAgent():
	'''
	Generate a randomized user agent by permuting a large set of possible values.
	The returned user agent should look like a valid, in-use brower, with a specified preferred language of english.

	Return value is a list of tuples, where each tuple is one of the user-agent headers.

	Currently can provide approximately 147 * 17 * 5 * 5 * 2 * 3 * 2 values, or ~749K possible
	unique user-agents.
	'''

	coding = random.choice(ENCODINGS)
	random.shuffle(coding)
	coding = random.choice((", ", ",")).join(coding)

	accept_list = [tmp for tmp in random.choice(ACCEPT)]
	accept_list.append(random.choice(ACCEPT_POSTFIX))
	accept_str = random.choice((", ", ",")).join(accept_list)

	assert accept_str.count("*.*") <= 1

	user_agent = [
				('User-Agent'		,	random.choice(USER_AGENTS)),
				('Accept-Language'	,	random.choice(ACCEPT_LANGUAGE)),
				('Accept'			,	accept_str),
				('Accept-Encoding'	,	coding)
				]
	return user_agent