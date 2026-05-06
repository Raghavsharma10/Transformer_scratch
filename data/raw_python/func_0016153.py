def read_bytes(self):
		"""
		reading bytes; update progress bar after 1 ms
		"""
		global exit_flag

		for self.i in range(0, self.length) :
			self.bytes[self.i] = i_max[self.i]
			self.maxbytes[self.i] = total_chunks[self.i]
			self.progress[self.i]["maximum"] = total_chunks[self.i]
			self.progress[self.i]["value"] = self.bytes[self.i]
			self.str[self.i].set(file_name[self.i]+ "       " + str(self.bytes[self.i]) 
								  + "KB / " + str(int(self.maxbytes[self.i] + 1)) + " KB")

		if exit_flag == self.length:
			exit_flag = 0
			self.frame.destroy()
		else:
			self.frame.after(10, self.read_bytes)