def log(**data):
	"""RPC method for logging events
	Makes entry with new account creating
	Return None
	"""
	# Get data from request body
	entry = {
		"module": data["params"]["module"],
		"event": data["params"]["event"],
		"timestamp": data["params"]["timestamp"],
		"arguments": data["params"]["arguments"]
	}
	# Call create metod for writing data to database
	history.create(entry)