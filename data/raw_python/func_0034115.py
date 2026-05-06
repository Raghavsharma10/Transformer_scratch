def checkfilesexist(self, on_missing="warn"):
		'''
		Runs through the entries of the Cache() object and checks each entry
		if the file which it points to exists or not. If the file does exist then 
		it adds the entry to the Cache() object containing found files, otherwise it
		adds the entry to the Cache() object containing all entries that are missing. 
		It returns both in the follwing order: Cache_Found, Cache_Missed.
		
		Pass on_missing to control how missing files are handled:
		  "warn": print a warning message saying how many files
		          are missing out of the total checked.
		  "error": raise an exception if any are missing
		  "ignore": do nothing
		'''  
		if on_missing not in ("warn", "error", "ignore"):
			raise ValueError("on_missing must be \"warn\", \"error\", or \"ignore\".")
		
		c_found = []
		c_missed = []
		for entry in self:
			if os.path.isfile(entry.path):
				c_found.append(entry)
			else:
				c_missed.append(entry)
		
		if len(c_missed) > 0:
			msg = "%d of %d files in the cache were not found "\
			    "on disk" % (len(c_missed), len(self))
			if on_missing == "warn":
				print >>sys.stderr, "warning: " + msg
			elif on_missing == "error":
				raise ValueError(msg)
			elif on_missing == "ignore":
				pass
			else:
				raise ValueError("Why am I here? "\
				      "Please file a bug report!")
		return self.__class__(c_found), self.__class__(c_missed)