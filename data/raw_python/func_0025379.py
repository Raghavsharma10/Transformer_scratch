def restore(self, workspace_uuid):
        """
            Restore the workspace to the given workspace_uuid.

            If workspace_uuid is None then create a new workspace and use it.
        """
        workspace = next((workspace for workspace in self.document_model.workspaces if workspace.uuid == workspace_uuid), None)
        if workspace is None:
            workspace = self.new_workspace()
        self._change_workspace(workspace)