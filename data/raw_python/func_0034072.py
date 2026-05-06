def reset_next_ids(classes):
	"""
	For each class in the list, if the .next_id attribute is not None
	(meaning the table has an ID generator associated with it), set
	.next_id to 0.  This has the effect of reseting the ID generators,
	and is useful in applications that process multiple documents and
	add new rows to tables in those documents.  Calling this function
	between documents prevents new row IDs from growing continuously
	from document to document.  There is no need to do this, it's
	purpose is merely aesthetic, but it can be confusing to open a
	document and find process ID 300 in the process table and wonder
	what happened to the other 299 processes.

	Example:

	>>> import lsctables
	>>> reset_next_ids(lsctables.TableByName.values())
	"""
	for cls in classes:
		if cls.next_id is not None:
			cls.set_next_id(type(cls.next_id)(0))