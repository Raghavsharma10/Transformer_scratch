def read_file(self):
        '''if this is stored in a file, read it into self.column'''
        column_selector = r'(.*)\[(\d+)\]$'
        if self.column_file:
            column = None
            m = re.match(column_selector,self.column_file)
            file = self.column_file
            if m:
                file = m.group(1)
                column = int(m.group(2))
            with open(file) as f:
                lines = f.read().split('\n')
                if column!=None:
                    lines = [x.split()[column] for x in lines]
                self.column = [nl.numberize(x) for x in lines]
            self.column_file = None
        if self.times_file:
            with open(self.times_file) as f:
                self.times = [[nl.numberize(x) for x in y.split()] for y in f.read().split('\n')]
            self.times_file = None