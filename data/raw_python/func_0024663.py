async def load(self):
        """Load scenes from KLF 200."""
        get_scene_list = GetSceneList(pyvlx=self.pyvlx)
        await get_scene_list.do_api_call()
        if not get_scene_list.success:
            raise PyVLXException("Unable to retrieve scene information")
        for scene in get_scene_list.scenes:
            self.add(Scene(pyvlx=self.pyvlx, scene_id=scene[0], name=scene[1]))