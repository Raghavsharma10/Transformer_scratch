def thumbnail_source_for_display_item(self, ui, display_item: DisplayItem.DisplayItem) -> ThumbnailSource:
        """Returned ThumbnailSource must be closed."""
        with self.__lock:
            thumbnail_source = self.__thumbnail_sources.get(display_item)
            if not thumbnail_source:
                thumbnail_source = ThumbnailSource(ui, display_item)
                self.__thumbnail_sources[display_item] = thumbnail_source

                def will_delete(thumbnail_source):
                    del self.__thumbnail_sources[thumbnail_source._display_item]

                thumbnail_source._on_will_delete = will_delete
            else:
                assert thumbnail_source._ui == ui
            return thumbnail_source.add_ref()