def fetch(url, binary, outfile, noprint, rendered):
	'''
	Fetch a specified URL's content, and output it to the console.
	'''
	with chrome_context.ChromeContext(binary=binary) as cr:
		resp = cr.blocking_navigate_and_get_source(url)
		if rendered:
			resp['content'] = cr.get_rendered_page_source()
			resp['binary'] = False
			resp['mimie'] = 'text/html'

	if not noprint:
		if resp['binary'] is False:
			print(resp['content'])
		else:
			print("Response is a binary file")
			print("Cannot print!")

	if outfile:
		with open(outfile, "wb") as fp:
			if resp['binary']:
				fp.write(resp['content'])
			else:
				fp.write(resp['content'].encode("UTF-8"))