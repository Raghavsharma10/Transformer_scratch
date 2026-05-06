def set(self, name, default=0, editable=True, description=""):
        '''Define a variable in DB and in memory'''

        var, created = ConfigurationVariable.objects.get_or_create(name=name)

        if created:
            var.value = default

        if not editable:
            var.value = default

        var.editable = editable
        var.description = description
        var.save(reload=False)

        # ATTRIBUTES is accesible by any instance of VariablesManager
        self.ATTRIBUTES[var.name] = var.value