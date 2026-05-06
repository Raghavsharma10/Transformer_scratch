def select_and_insert(self, name, data):
        '''Combines selection and data insertion into one function
        
        Args:
            name: the name of the object you want to insert into
            data: the data you want to insert
        Returns:
            None
        Raises:
            None
        '''
        self.select_obj(name)
        self.insert_into_obj(data)