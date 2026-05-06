def clear(self):
		'''
		claar all the cache, and release memory
		'''
		for node in self.dli():
			node.empty = True
			node.key = None
			node.value = None
		
		self.head = _dlnode()
		self.head.next = self.head
		self.head.prev = self.head
		self.listSize = 1
		
		self.table.clear()
		
		# status var
		self.hit_cnt = 0 
		self.miss_cnt = 0
		self.remove_cnt = 0