def read(self):
        '''Read some number of messages'''
        found = Client.read(self)

        # Redistribute our ready state if necessary
        if self.needs_distribute_ready():
            self.distribute_ready()

        # Finally, return all the results we've read
        return found