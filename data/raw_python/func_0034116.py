def get_ilwdchar_class(tbl_name, col_name, namespace = globals()):
	"""
	Searches this module's namespace for a subclass of _ilwd.ilwdchar
	whose table_name and column_name attributes match those provided.
	If a matching subclass is found it is returned; otherwise a new
	class is defined, added to this module's namespace, and returned.

	Example:

	>>> process_id = get_ilwdchar_class("process", "process_id")
	>>> x = process_id(10)
	>>> str(type(x))
	"<class 'pycbc_glue.ligolw.ilwd.process_process_id_class'>"
	>>> str(x)
	'process:process_id:10'

	Retrieving and storing the class provides a convenient mechanism
	for quickly constructing new ID objects.

	Example:

	>>> for i in range(10):
	...	print str(process_id(i))
	...
	process:process_id:0
	process:process_id:1
	process:process_id:2
	process:process_id:3
	process:process_id:4
	process:process_id:5
	process:process_id:6
	process:process_id:7
	process:process_id:8
	process:process_id:9
	"""
	#
	# if the class already exists, retrieve and return it
	#

	key = (str(tbl_name), str(col_name))
	cls_name = "%s_%s_class" % key
	assert cls_name != "get_ilwdchar_class"
	try:
		return namespace[cls_name]
	except KeyError:
		pass

	#
	# otherwise define a new class, and add it to the cache
	#

	class new_class(_ilwd.ilwdchar):
		__slots__ = ()
		table_name, column_name = key
		index_offset = len("%s:%s:" % key)

	new_class.__name__ = cls_name

	namespace[cls_name] = new_class

	#
	# pickle support
	#

	copy_reg.pickle(new_class, lambda x: (ilwdchar, (unicode(x),)))

	#
	# return the new class
	#

	return new_class