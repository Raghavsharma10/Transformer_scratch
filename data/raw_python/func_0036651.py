def bdecode(f_or_data):
	"""
	bdecodes data by looking up the type byte,
	and using it to look up the respective decoding function,
	which in turn is used to return the decoded object
	
	The parameter can be a file opened in bytes mode,
	bytes or a string (the last of which will be decoded)
	"""
	if isinstance(f_or_data, str):
		f_or_data = f_or_data.encode()
	if isinstance(f_or_data, bytes):
		f_or_data = BytesIO(f_or_data)
	
	#TODO: the following line is the only one that needs readahead.
	#peek returns a arbitrary amount of bytes, so we have to slice.
	if f_or_data.seekable():
		first_byte = f_or_data.read(1)
		f_or_data.seek(-1, SEEK_CUR)
	else:
		first_byte = f_or_data.peek(1)[:1]
	btype = TYPES.get(first_byte)
	if btype is not None:
		return btype(f_or_data)
	else: #Used in dicts and lists to designate an end
		assert_btype(f_or_data.read(1), _TYPE_END)
		return None