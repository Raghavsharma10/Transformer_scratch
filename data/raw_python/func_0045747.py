def set_agnocomplete(self, klass_or_instance, user):
        """
        Handling the assignation of the agnocomplete object inside the field.
        A developer may want to use a class or an instance of an
        :class:`AgnocompleteBase` class to configure her field.

        Ex::

            from agnocomplete import Fields

            class SearchForm(forms.Form):
                search_class = fields.AgnocompleteField(AgnocompleteColor)
                search_class2 = fields.AgnocompleteField('AgnocompleteColor')
                search_instance = fields.AgnocompleteField(
                    AgnocompleteColor(page_size=3))

        if it's a :class: being passed as a parameter, it'll be
        instantiated using the default parameters. If it's a string, it'll
        be instanciated also, using the name of the class as the key to
        fetch the actual class.

        """
        # If string, use register to fetch the class
        if isinstance(klass_or_instance, six.string_types):
            registry = get_agnocomplete_registry()
            if klass_or_instance not in registry:
                raise UnregisteredAgnocompleteException(
                    "Unregistered Agnocomplete class: {} is unknown".format(klass_or_instance)  # noqa
                )
            klass_or_instance = registry[klass_or_instance]
        # If not an instance, instanciate this
        if not isinstance(klass_or_instance, AgnocompleteBase):
            klass_or_instance = klass_or_instance(user=user)
        # Pass the field when we have an AgnocompleteBase instance
        if isinstance(klass_or_instance, AgnocompleteBase):
            klass_or_instance.set_agnocomplete_field(self)
        # Store it in the instance
        self.agnocomplete = klass_or_instance
        self.agnocomplete.user = user