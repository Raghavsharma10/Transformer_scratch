def extract(connection, filename, table_names = None, verbose = False, xsl_file = None):
	"""
	Convert the database at the given connection to a tabular LIGO
	Light-Weight XML document.  The XML document is written to the file
	named filename.  If table_names is not None, it should be a
	sequence of strings and only the tables in that sequence will be
	converted.  If verbose is True then progress messages will be
	printed to stderr.
	"""
	xmldoc = ligolw.Document()
	xmldoc.appendChild(dbtables.get_xml(connection, table_names))
	ligolw_utils.write_filename(xmldoc, filename, gz = (filename or "stdout").endswith(".gz"), verbose = verbose, xsl_file = xsl_file)

	# delete cursors
	xmldoc.unlink()