def file_follow_durable( path,
		min_dump_interval=10,
		xattr_name='user.collectd.logtail.pos', xattr_update=True,
		**follow_kwz ):
	'''Records log position into xattrs after reading line every
			min_dump_interval seconds.
		Checksum of the last line at the position
			is also recorded (so line itself don't have to fit into xattr) to make sure
			file wasn't truncated between last xattr dump and re-open.'''

	from xattr import xattr
	from io import open
	from hashlib import sha1
	from time import time
	import struct

	# Try to restore position
	src = open(path, mode='rb')
	src_xattr = xattr(src)
	try:
		if not xattr_name: raise KeyError
		pos = src_xattr[xattr_name]
	except KeyError: pos = None
	if pos:
		data_len = struct.calcsize('=I')
		(pos,), chksum = struct.unpack('=I', pos[:data_len]), pos[data_len:]
		(data_len,), chksum = struct.unpack('=I', chksum[:data_len]), chksum[data_len:]
		try:
			src.seek(pos - data_len)
			if sha1(src.read(data_len)).digest() != chksum:
				raise IOError('Last log line doesnt match checksum')
		except (OSError, IOError) as err:
			collectd.info('Failed to restore log position: {}'.format(err))
			src.seek(0)
	tailer = file_follow(src, yield_file=True, **follow_kwz)

	# ...and keep it updated
	pos_dump_ts_get = lambda ts=None: (ts or time()) + min_dump_interval
	pos_dump_ts = pos_dump_ts_get()
	while True:
		line, src_chk = next(tailer)
		if not line: pos_dump_ts = 0 # force-write xattr
		ts = time()
		if ts > pos_dump_ts:
			if src is not src_chk:
				src, src_xattr = src_chk, xattr(src_chk)
			pos_new = src.tell()
			if pos != pos_new:
				pos = pos_new
				if xattr_update:
					src_xattr[xattr_name] =\
						struct.pack('=I', pos)\
						+ struct.pack('=I', len(line))\
						+ sha1(line).digest()
			pos_dump_ts = pos_dump_ts_get(ts)
		if (yield line.decode('utf-8', 'replace')):
			tailer.send(StopIteration)
			break