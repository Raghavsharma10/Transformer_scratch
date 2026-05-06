def refresh(self):
        """Refresh the data used by get_value.

        Only needed if you're not using subscriptions.
        """
        j = self.vera_request(id='sdata', output_format='json').json()
        scenes = j.get('scenes')
        for scene_data in scenes:
            if scene_data.get('id') == self.scene_id:
                self.update(scene_data)