async def run(self, wait_for_completion=True):
        """Run scene.

        Parameters:
            * wait_for_completion: If set, function will return
                after device has reached target position.

        """
        activate_scene = ActivateScene(
            pyvlx=self.pyvlx,
            wait_for_completion=wait_for_completion,
            scene_id=self.scene_id)
        await activate_scene.do_api_call()
        if not activate_scene.success:
            raise PyVLXException("Unable to activate scene")