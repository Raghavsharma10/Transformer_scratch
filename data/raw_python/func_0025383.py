def rename_workspace(self) -> None:
        """ Pose a dialog to rename the workspace. """

        def rename_clicked(text):
            if len(text) > 0:
                command = Workspace.RenameWorkspaceCommand(self, text)
                command.perform()
                self.document_controller.push_undo_command(command)

        self.pose_get_string_message_box(caption=_("Enter new name for workspace"), text=self.__workspace.name,
                                         accepted_fn=rename_clicked, accepted_text=_("Rename"),
                                         message_box_id="rename_workspace")