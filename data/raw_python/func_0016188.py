def get_url_nofollow(url):
	""" 
	function to get return code of a url

	Credits: http://blog.jasonantman.com/2013/06/python-script-to-check-a-list-of-urls-for-return-code-and-final-return-code-if-redirected/
	"""
	try:
		response = urlopen(url)
		code = response.getcode()
		return code
	except HTTPError as e:
		return e.code
	except:
		return 0