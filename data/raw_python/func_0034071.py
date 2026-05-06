def reassign_ids(elem):
	"""
	Recurses over all Table elements below elem whose next_id
	attributes are not None, and uses the .get_next_id() method of each
	of those Tables to generate and assign new IDs to their rows.  The
	modifications are recorded, and finally all ID attributes in all
	rows of all tables are updated to fix cross references to the
	modified IDs.

	This function is used by ligolw_add to assign new IDs to rows when
	merging documents in order to make sure there are no ID collisions.
	Using this function in this way requires the .get_next_id() methods
	of all Table elements to yield unused IDs, otherwise collisions
	will result anyway.  See the .sync_next_id() method of the Table
	class for a way to initialize the .next_id attributes so that
	collisions will not occur.

	Example:

	>>> import ligolw
	>>> import lsctables
	>>> xmldoc = ligolw.Document()
	>>> xmldoc.appendChild(ligolw.LIGO_LW()).appendChild(lsctables.New(lsctables.SnglInspiralTable))
	[]
	>>> reassign_ids(xmldoc)
	"""
	mapping = {}
	for tbl in elem.getElementsByTagName(ligolw.Table.tagName):
		if tbl.next_id is not None:
			tbl.updateKeyMapping(mapping)
	for tbl in elem.getElementsByTagName(ligolw.Table.tagName):
		tbl.applyKeyMapping(mapping)