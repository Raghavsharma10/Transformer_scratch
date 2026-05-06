def ask_for_input_with_prompt(cls, ui, prompt='', **options):
        """Ask user for written input with prompt"""
        return cls.get_appropriate_helper(ui).ask_for_input_with_prompt(prompt=prompt, **options)