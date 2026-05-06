def send_data(self, data):
        ''' This function can be overwritten in derived class

            Std. function is to broadcast all receiver data to all backends
        '''
        for frontend_data in data:
            serialized_data = self.serialize_data(frontend_data)
            if sys.version_info >= (3, 0):
                serialized_data = serialized_data.encode('utf-8')
            for actual_backend in self.backends:
                actual_backend[1].send(serialized_data)