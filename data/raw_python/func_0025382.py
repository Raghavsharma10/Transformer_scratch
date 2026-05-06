def create_workspace(self) -> None:
        """ Pose a dialog to name and create a workspace. """

        def create_clicked(text):
            if text:
                command = Workspace.CreateWorkspaceCommand(self, text)
                command.perform()
                self.document_controller.push_undo_command(command)

        self.pose_get_string_message_box(caption=_("Enter a name for the workspace"), text=_("Workspace"),
                                         accepted_fn=create_clicked, accepted_text=_("Create"),
                                         message_box_id="create_workspace")