def ensure_workspace(self, name, layout, workspace_id):
        """Looks for a workspace with workspace_id.

        If none is found, create a new one, add it, and change to it.
        """
        workspace = next((workspace for workspace in self.document_model.workspaces if workspace.workspace_id == workspace_id), None)
        if not workspace:
            workspace = self.new_workspace(name=name, layout=layout, workspace_id=workspace_id)
        self._change_workspace(workspace)