def _get_two_data_sources(self):
        """Get two sensible data sources, which may be the same."""
        selected_display_items = self.selected_display_items
        if len(selected_display_items) < 2:
            selected_display_items = list()
            display_item = self.selected_display_item
            if display_item:
                selected_display_items.append(display_item)
        if len(selected_display_items) == 1:
            display_item = selected_display_items[0]
            data_item = display_item.data_item if display_item else None
            if display_item and len(display_item.graphic_selection.indexes) == 2:
                index1 = display_item.graphic_selection.anchor_index
                index2 = list(display_item.graphic_selection.indexes.difference({index1}))[0]
                graphic1 = display_item.graphics[index1]
                graphic2 = display_item.graphics[index2]
                if data_item:
                    if data_item.is_datum_1d and isinstance(graphic1, Graphics.IntervalGraphic) and isinstance(graphic2, Graphics.IntervalGraphic):
                        crop_graphic1 = graphic1
                        crop_graphic2 = graphic2
                    elif data_item.is_datum_2d and isinstance(graphic1, Graphics.RectangleTypeGraphic) and isinstance(graphic2, Graphics.RectangleTypeGraphic):
                        crop_graphic1 = graphic1
                        crop_graphic2 = graphic2
                    else:
                        crop_graphic1 = self.__get_crop_graphic(display_item)
                        crop_graphic2 = crop_graphic1
                else:
                    crop_graphic1 = self.__get_crop_graphic(display_item)
                    crop_graphic2 = crop_graphic1
            else:
                crop_graphic1 = self.__get_crop_graphic(display_item)
                crop_graphic2 = crop_graphic1
            return (display_item, crop_graphic1), (display_item, crop_graphic2)
        if len(selected_display_items) == 2:
            display_item1 = selected_display_items[0]
            crop_graphic1 = self.__get_crop_graphic(display_item1)
            display_item2 = selected_display_items[1]
            crop_graphic2 = self.__get_crop_graphic(display_item2)
            return (display_item1, crop_graphic1), (display_item2, crop_graphic2)
        return None