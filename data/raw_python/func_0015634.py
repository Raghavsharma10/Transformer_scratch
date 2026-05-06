def add_actions(self, entries, user_data=None):
        """
        The add_actions() method is a convenience method that creates a number
        of gtk.Action  objects based on the information in the list of action
        entry tuples contained in entries and adds them to the action group.
        The entry tuples can vary in size from one to six items with the
        following information:

            * The name of the action. Must be specified.
            * The stock id for the action. Optional with a default value of None
              if a label is specified.
            * The label for the action. This field should typically be marked
              for translation, see the set_translation_domain() method. Optional
              with a default value of None if a stock id is specified.
            * The accelerator for the action, in the format understood by the
              gtk.accelerator_parse() function. Optional with a default value of
              None.
            * The tooltip for the action. This field should typically be marked
              for translation, see the set_translation_domain() method. Optional
              with a default value of None.
            * The callback function invoked when the action is activated.
              Optional with a default value of None.

        The "activate" signals of the actions are connected to the callbacks and
        their accel paths are set to <Actions>/group-name/action-name.
        """
        try:
            iter(entries)
        except (TypeError):
            raise TypeError('entries must be iterable')

        def _process_action(name, stock_id=None, label=None, accelerator=None, tooltip=None, callback=None):
            action = Action(name=name, label=label, tooltip=tooltip, stock_id=stock_id)
            if callback is not None:
                if user_data is None:
                    action.connect('activate', callback)
                else:
                    action.connect('activate', callback, user_data)

            self.add_action_with_accel(action, accelerator)

        for e in entries:
            # using inner function above since entries can leave out optional arguments
            _process_action(*e)