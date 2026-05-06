def list_line(self, line):
        """
        Write the given iterable of values (line) to the file as items on the 
        same line. Any argument that stringifies to a string legal as a TSV data
        item can be written.
        
        Does not copy the line or build a big string in memory.
        """
        
        if len(line) == 0:
            return
        
        self.stream.write(str(line[0]))
        
        for item in line[1:]:
            self.stream.write("\t")
            self.stream.write(str(item))
        
        self.stream.write("\n")