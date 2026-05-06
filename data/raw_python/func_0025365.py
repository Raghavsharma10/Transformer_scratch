def graphviz(self, directory, **kwargs):
        "Directory to store the graphviz models"
        import os
        if not os.path.isdir(directory):
            os.mkdir(directory)
        output = os.path.join(directory, 'evodag-%s')
        for k, m in enumerate(self.models):
            m.graphviz(output % k, **kwargs)