def clone_workspace(self) -> None:
        """ Pose a dialog to name and clone a workspace. """

        def clone_clicked(text):
            if text:
                command = Workspace.CloneWorkspaceCommand(self, text)
                command.perform()
                self.document_controller.push_undo_command(command)

        self.pose_get_string_message_box(caption=_("Enter a name for the workspace"), text=self.__workspace.name,
                                         accepted_fn=clone_clicked, accepted_text=_("Clone"),
                                         message_box_id="clone_workspace")