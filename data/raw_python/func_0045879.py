def finish(self):
        """Declare this section finished"""
        self._my_map['over'] = True  # finished == over?
        self._my_map['completionTime'] = DateTime.utcnow()
        self._save()