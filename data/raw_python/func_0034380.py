def getquery(query):
	'Performs a query and get the results.'
	try:
		conn = connection.cursor()
		conn.execute(query)
		data = conn.fetchall()
		conn.close()
	except: data = list()
	return data