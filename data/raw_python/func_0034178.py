def write_fileobj(xmldoc, fileobj, gz = False, trap_signals = (signal.SIGTERM, signal.SIGTSTP), **kwargs):
	"""
	Writes the LIGO Light Weight document tree rooted at xmldoc to the
	given file object.  Internally, the .write() method of the xmldoc
	object is invoked and any additional keyword arguments are passed
	to that method.  The file object need not be seekable.  The output
	data is gzip compressed on the fly if gz is True.  The return value
	is a string containing the hex digits of the MD5 digest of the
	output bytestream.

	This function traps the signals in the trap_signals iterable during
	the write process (the default is signal.SIGTERM and
	signal.SIGTSTP), and it does this by temporarily installing its own
	signal handlers in place of the current handlers.  This is done to
	prevent Condor eviction during the write process.  When the file
	write is concluded the original signal handlers are restored.
	Then, if signals were trapped during the write process, the signals
	are then resent to the current process in the order in which they
	were received.  The signal.signal() system call cannot be invoked
	from threads, and trap_signals must be set to None or an empty
	sequence if this function is used from a thread.

	Example:

	>>> import sys
	>>> from pycbc_glue.ligolw import ligolw
	>>> xmldoc = load_filename("demo.xml", contenthandler = ligolw.LIGOLWContentHandler)
	>>> digest = write_fileobj(xmldoc, sys.stdout)	# doctest: +NORMALIZE_WHITESPACE
	<?xml version='1.0' encoding='utf-8'?>
	<!DOCTYPE LIGO_LW SYSTEM "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt">
	<LIGO_LW>
		<Table Name="demo:table">
			<Column Type="lstring" Name="name"/>
			<Column Type="real8" Name="value"/>
			<Stream Delimiter="," Type="Local" Name="demo:table">
	"mass",0.5,"velocity",34
			</Stream>
		</Table>
	</LIGO_LW>
	>>> digest
	'37044d979a79409b3d782da126636f53'
	"""
	# initialize SIGTERM and SIGTSTP trap
	deferred_signals = []
	def newsigterm(signum, frame):
		deferred_signals.append(signum)
	oldhandlers = {}
	if trap_signals is not None:
		for sig in trap_signals:
			oldhandlers[sig] = signal.getsignal(sig)
			signal.signal(sig, newsigterm)

	# write the document
	with MD5File(fileobj, closable = False) as fileobj:
		md5obj = fileobj.md5obj
		with fileobj if not gz else GzipFile(mode = "wb", fileobj = fileobj) as fileobj:
			with codecs.getwriter("utf_8")(fileobj) as fileobj:
				xmldoc.write(fileobj, **kwargs)

	# restore original handlers, and send outselves any trapped signals
	# in order
	for sig, oldhandler in oldhandlers.iteritems():
		signal.signal(sig, oldhandler)
	while deferred_signals:
		os.kill(os.getpid(), deferred_signals.pop(0))

	# return the hex digest of the bytestream that was written
	return md5obj.hexdigest()