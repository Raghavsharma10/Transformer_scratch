def get_multiple_choices_required(self):
        """
        Add only the required message, but no 'ng-required' attribute to the input fields,
        otherwise all Checkboxes of a MultipleChoiceField would require the property "checked".
        """
        errors = []
        if self.required:
            for key, msg in self.error_messages.items():
                if key == 'required':
                    errors.append(('$error.required', msg))
        return errors