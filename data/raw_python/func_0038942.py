def get_id(self):
        """Returns unique id of an alignment.  """
        return hash(str(self.title) + str(self.best_score()) + str(self.hit_def))