def main(args=None):
	"""Decodes bencoded files to python syntax (like JSON, but with bytes support)"""
	import sys, pprint
	from argparse import ArgumentParser, FileType
	parser = ArgumentParser(description=main.__doc__)
	parser.add_argument('infile',  nargs='?', type=FileType('rb'), default=sys.stdin.buffer,
		help='bencoded file (e.g. torrent) [Default: stdin]')
	parser.add_argument('outfile', nargs='?', type=FileType('w'), default=sys.stdout,
		help='python-syntax serialization [Default: stdout]')
	args = parser.parse_args(args)
	
	data = bdecode(args.infile)
	pprint.pprint(data, stream=args.outfile)