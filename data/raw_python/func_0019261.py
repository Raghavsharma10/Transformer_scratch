def listofmodeluserfunctions(self):
        """User functions of the model class."""
        lines = []
        for (name, member) in vars(self.model.__class__).items():
            if (inspect.isfunction(member) and
                    (name not in ('run', 'new2old')) and
                    ('fastaccess' in inspect.getsource(member))):
                lines.append((name, member))
        run = vars(self.model.__class__).get('run')
        if run is not None:
            lines.append(('run', run))
        for (name, member) in vars(self.model).items():
            if (inspect.ismethod(member) and
                    ('fastaccess' in inspect.getsource(member))):
                lines.append((name, member))
        return lines