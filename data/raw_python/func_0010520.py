def get_child_progress(self, parent_min, parent_max):
        """
        Create a new child ProgressCallback.
        Minimum and maximum values of the child are mapped to parent_min and parent_max of this parent ProgressCallback.
        :param parent_min: minimum value of the child is mapped to parent_min of this parent ProgressCallback
        :param parent_max: maximum value of the child is mapped to parent_max of this parent ProgressCallback
        :return: instance of SubProgressCallback
        """
        return SubProgressCallback(parent=self, parent_min=parent_min, parent_max=parent_max)