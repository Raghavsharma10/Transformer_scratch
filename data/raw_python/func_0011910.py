def course(self):
        """
        Course this node belongs to
        """
        course = self.parent
        while course.parent:
            course = course.parent
        return course