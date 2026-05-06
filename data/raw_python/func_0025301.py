def show_get_string_message_box(self, caption: str, text: str, accepted_fn, rejected_fn=None, accepted_text: str=None, rejected_text: str=None) -> None:
        """Show a dialog box and ask for a string.

        Caption describes the user prompt. Text is the initial/default string.

        Accepted function must be a function taking one argument which is the resulting text if the user accepts the
        message dialog. It will only be called if the user clicks OK.

        Rejected function can be a function taking no arguments, called if the user clicks Cancel.

        .. versionadded:: 1.0

        Scriptable: No
        """
        workspace = self.__document_controller.workspace_controller
        workspace.pose_get_string_message_box(caption, text, accepted_fn, rejected_fn, accepted_text, rejected_text)