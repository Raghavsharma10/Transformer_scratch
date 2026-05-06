def ask_for_confirm_with_message(cls, ui, prompt='Do you agree?', message='', **options):
        """Returns True if user agrees, False otherwise"""
        return cls.get_appropriate_helper(ui).ask_for_confirm_with_message(prompt, message)