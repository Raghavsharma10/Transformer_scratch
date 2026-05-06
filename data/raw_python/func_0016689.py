def clean_image(self):
        """
        It seems like in Django 1.5 something has changed.

        When Django tries to validate the form, it checks if the generated
        filename fit into the max_length. But at this point, self.instance.user
        is not yet set so our filename generation function cannot create
        the new file path because it needs the user id. Setting
        self.instance.user at this point seems to work as a workaround.

        """
        self.instance.user = self.user
        data = self.cleaned_data.get('image')
        return data