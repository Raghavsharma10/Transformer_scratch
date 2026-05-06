def cli(verbose, silent):
	'''
ChromeController

\b
Usage: python3 -m ChromeController [-s | --silent] [-v | --verbose]
  python3 -m ChromeController fetch <url> [--binary <bin_name>] [--outfile <out_file_name>]
  python3 -m ChromeController update
  python3 -m ChromeController (-h | --help)
  python3 -m ChromeController --version

\b
Options:
  -s --silent   Suppress all output aside from the fetched content
                This basically makes ChromeController act like a alternative to curl
  -v --verbose  The opposite of silent. Causes the internal logging to output
                all traffic over the chromium control interface. VERY noisy.
  --version     Show version.
  fetch         Fetch a specified URL's content, and output it to the console.
	'''

	if verbose:
		logging.basicConfig(level=logging.DEBUG)
	elif silent:
		logging.basicConfig(level=logging.ERROR)
	else:
		logging.basicConfig(level=logging.INFO)