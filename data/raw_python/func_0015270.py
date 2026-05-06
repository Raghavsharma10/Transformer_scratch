def btn_press_event(self, widget, event):
        """
        Function is used for showing Popup menu
        """
        if event.type == Gdk.EventType.BUTTON_PRESS:
            if event.button.button == 1:
                widget.popup(None, None, None, None,
                             event.button.button, event.time)
            return True
        return False