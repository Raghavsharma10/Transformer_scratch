def partition(self):
        """Partitions all tasks into groups of tasks. A group is
           represented by a task_store object that indexes a sub-
           set of tasks."""
        step = int(math.ceil(self.num_tasks / float(self.partitions)))
        if self.indices == None:
            slice_ind = list(range(0, self.num_tasks, step))
            for start in slice_ind:
                yield self.__class__(self.partitions, 
                                     list(range(start, start + step)))
        else:
            slice_ind = list(range(0, len(self.indices), step))
            for start in slice_ind:
                if start + step <= len(self.indices):
                    yield self.__class__(self.partitions, 
                                         self.indices[start: start + step])
                else:
                    yield self.__class__(self.partitions, self.indices[start:])