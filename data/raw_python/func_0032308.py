def machine_op(self, operation):
        '''Perform machine operations
        
        Args:
            operations: which operation you would like
        Returns:
            None
        Raises:
            RuntimeError: Invalid operation
        '''
        operations = {'feed2start': 1,
                      'feedone': 2,
                      'cut': 3
                      }
        
        if operation in operations:
            self.send('^'+'O'+'P'+chr(operations[operation]))
        else:
            raise RuntimeError('Invalid operation.')