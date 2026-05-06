def vote(seglists, n):
	"""
	Given a sequence of segmentlists, returns the intervals during
	which at least n of them intersect.  The input segmentlists must be
	coalesced, the output is coalesced.

	Example:

	>>> from pycbc_glue.segments import *
	>>> w = segmentlist([segment(0, 15)])
	>>> x = segmentlist([segment(5, 20)])
	>>> y = segmentlist([segment(10, 25)])
	>>> z = segmentlist([segment(15, 30)])
	>>> vote((w, x, y, z), 3)
	[segment(10, 20)]

	The sequence of segmentlists is only iterated over once, and the
	segmentlists within it are only iterated over once;  they can all
	be generators.  If there are a total of N segments in M segment
	lists and the final result has L segments the algorithm is O(N M) +
	O(L).
	"""
	# check for no-op

	if n < 1:
		return segments.segmentlist()

	# digest the segmentlists into an ordered sequence of off-on and
	# on-off transitions with the vote count for each transition
	# FIXME:  this generator is declared locally for now, is it useful
	# as a stand-alone generator?

	def pop_min(l):
		# remove and return the smallest value from a list
		val = min(l)
		for i in xrange(len(l) - 1, -1, -1):
			if l[i] is val:
				return l.pop(i)
		assert False	# cannot get here

	def vote_generator(seglists):
		queue = []
		for seglist in seglists:
			segiter = iter(seglist)
			try:
				seg = segiter.next()
			except StopIteration:
				continue
			# put them in so that the smallest boundary is
			# closest to the end of the list
			queue.append((seg[1], -1, segiter))
			queue.append((seg[0], +1, None))
		if not queue:
			return
		queue.sort(reverse = True)
		bound = queue[-1][0]
		votes = 0
		while queue:
			this_bound, delta, segiter = pop_min(queue)
			if this_bound == bound:
				votes += delta
			else:
				yield bound, votes
				bound = this_bound
				votes = delta
			if segiter is not None:
				try:
					seg = segiter.next()
				except StopIteration:
					continue
				queue.append((seg[1], -1, segiter))
				queue.append((seg[0], +1, None))
		yield bound, votes

	# compute the cumulative sum of votes, and assemble a segmentlist
	# from the intervals when the vote count is equal to or greater
	# than n

	result = segments.segmentlist()
	votes = 0
	for bound, delta in vote_generator(seglists):
		if delta > 0 and n - delta <= votes < n:
			start = bound
		elif delta < 0 and n <= votes < n - delta:
			result.append(segments.segment(start, bound))
			del start	# detect stops that aren't preceded by starts
		votes += delta
	assert votes == 0	# detect failed cumulative sum

	return result