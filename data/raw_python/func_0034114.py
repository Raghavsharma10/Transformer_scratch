def sieve(self, ifos=None, description=None, segment=None,
		segmentlist=None, exact_match=False):
		"""
		Return a Cache object with those CacheEntries that
		contain the given patterns (or overlap, in the case of
		segment or segmentlist).  If exact_match is True, then
		non-None ifos, description, and segment patterns must
		match exactly, and a non-None segmentlist must contain
		a segment which matches exactly).

		It makes little sense to specify both segment and
		segmentlist arguments, but it is not prohibited.

		Bash-style wildcards (*?) are allowed for ifos and description.
		"""
		if exact_match:
			segment_func = lambda e: e.segment == segment
			segmentlist_func = lambda e: e.segment in segmentlist
		else:
			if ifos is not None: ifos = "*" + ifos + "*"
			if description is not None: description = "*" + description + "*"
			segment_func = lambda e: segment.intersects(e.segment)
			segmentlist_func = lambda e: segmentlist.intersects_segment(e.segment)
		
		c = self
		
		if ifos is not None:
			ifos_regexp = re.compile(fnmatch.translate(ifos))
			c = [entry for entry in c if ifos_regexp.match(entry.observatory) is not None]
		
		if description is not None:
			descr_regexp = re.compile(fnmatch.translate(description))
			c = [entry for entry in c if descr_regexp.match(entry.description) is not None]
		
		if segment is not None:
			c = [entry for entry in c if segment_func(entry)]
		
		if segmentlist is not None:
			# must coalesce for intersects_segment() to work
			segmentlist.coalesce()
			c = [entry for entry in c if segmentlist_func(entry)]

		return self.__class__(c)