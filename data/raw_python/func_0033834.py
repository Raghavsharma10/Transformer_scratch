def from_bitstream(bitstream, start, dt, minlen = 1):
	"""
	Convert consecutive True values in a bit stream (boolean-castable
	iterable) to a stream of segments. Require minlen consecutive True
	samples to comprise a segment.

	Example:

	>>> list(from_bitstream((True, True, False, True, False), 0, 1))
	[segment(0, 2), segment(3, 4)]
	>>> list(from_bitstream([[], [[]], [[]], [], []], 1013968613, 0.125))
	[segment(1013968613.125, 1013968613.375)]
	"""
	bitstream = iter(bitstream)
	i = 0
	while 1:
		if bitstream.next():
			# found start of True block; find the end
			j = i + 1
			try:
				while bitstream.next():
					j += 1
			finally:  # make sure StopIteration doesn't kill final segment
				if j - i >= minlen:
					yield segments.segment(start + i * dt, start + j * dt)
			i = j  # advance to end of block
		i += 1