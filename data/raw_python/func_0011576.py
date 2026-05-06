def question_default_loader(self, pk):
        """Load a Question from the database."""
        try:
            obj = Question.objects.get(pk=pk)
        except Question.DoesNotExist:
            return None
        else:
            self.question_default_add_related_pks(obj)
            return obj