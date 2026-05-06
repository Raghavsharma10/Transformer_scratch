def get_or_create_by_content(cls, name, content):
        """
        if xml content updated, create a new entry for given wf name
        Args:
            name: name of wf
            content: xml content

        Returns (DiagramXML(), bool): A tuple with two members.
        (DiagramXML instance and True if it's new or False it's already exists
        """
        new = False
        diagrams = cls.objects.filter(name=name)
        if diagrams:
            diagram = diagrams[0]
            if diagram.body != content:
                new = True
        else:
            new = True
        if new:
            diagram = cls(name=name, body=content).save()
        return diagram, new