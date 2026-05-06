def voxel_loop(self):
        '''iterator that loops through each voxel and yields the coords and time series as a tuple'''
        # Prob not the most efficient, but the best I can do for now:
        for x in xrange(len(self.data)):
            for y in xrange(len(self.data[x])):
                for z in xrange(len(self.data[x][y])):
                    yield ((x,y,z),self.data[x][y][z])