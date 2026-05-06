def open_(filename, mode=None, compresslevel=9):
   """Switch for both open() and gzip.open().
   
   Determines if the file is normal or gzipped by looking at the file
   extension.
   
   The filename argument is required; mode defaults to 'rb' for gzip and 'r'
   for normal and compresslevel defaults to 9 for gzip.
   
   >>> import gzip
   >>> from contextlib import closing
   >>> with open_(filename) as f:
   ...     f.read()
   """
   if filename[-3:] == '.gz':
      if mode is None: mode = 'rt'
      return closing(gzip.open(filename, mode, compresslevel))
   else:
      if mode is None: mode = 'r'
      return open(filename, mode)