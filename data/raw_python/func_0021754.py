def unfold_subtitle_stub(self, subtitle_stub):
        """Turn a SubtitleStub into a full Subtitle object

        @param crunchyroll.models.SubtitleStub subtitle_stub
        @return crunchyroll.models.Subtitle
        """
        return Subtitle(self._ajax_api.Subtitle_GetXml(
            subtitle_script_id=int(subtitle_stub.id)))