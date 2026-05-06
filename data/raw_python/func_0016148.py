def download_parallel(url, directory, idx, min_file_size = 0, max_file_size = -1,  
			 no_redirects = False, pos = 0, mode = 's'):
	"""
	download function to download parallely
	"""
	global main_it
	global exit_flag
	global total_chunks
	global file_name
	global i_max

	file_name[idx]= url.split('/')[-1] 
	file_address = directory + '/' + file_name[idx]
	is_redirects = not no_redirects

	resp = s.get(url, stream = True, allow_redirects = is_redirects)
	if not resp.status_code == 200:
		# ignore this file since server returns invalid response
		exit_flag += 1
		return
	try:
		total_size = int(resp.headers['content-length'])
	except KeyError:
		total_size = len(resp.content)

	total_chunks[idx] = total_size / chunk_size
	if total_chunks[idx] < min_file_size: 
		# ignore this file since file size is lesser than min_file_size
		exit_flag += 1
		return
	elif max_file_size != -1 and total_chunks[idx] > max_file_size:
		# ignore this file since file size is greater than max_file_size
		exit_flag += 1
		return

	file_iterable = resp.iter_content(chunk_size = chunk_size)
	with open(file_address, 'wb') as f:
		for sno, data in enumerate(file_iterable):
			i_max[idx] = sno + 1
			f.write(data)
	
	exit_flag += 1