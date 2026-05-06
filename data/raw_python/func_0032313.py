def insert_into_obj(self, data):
        '''Insert text into selected object.
        
        Args:
            data: The data you want to insert.
        Returns:
            None
        Raises:
            None
        '''
        if not data:
            data = ''
        size = len(data)
        n1 = size%256
        n2 = size/256
            
        self.send('^DI'+chr(n1)+chr(n2)+data)