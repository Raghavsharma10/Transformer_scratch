def make_image_converters(self, image, configuration, harpoon_spec):
        """Make converters for this image and add them to the configuration"""
        def convert_image(path, val):
            log.info("Converting %s", path)
            everything = path.configuration.root().wrapped()
            meta = Meta(everything, [])
            configuration.converters.started(path)

            base = path.configuration.root().wrapped()
            base.update(configuration.as_dict(ignore=["images"]))
            base.update(val.as_dict(ignore=["images"]))

            base["__image__"] = base
            everything["__image__"] = base

            base["harpoon"] = configuration["harpoon"]
            base["configuration"] = configuration
            return harpoon_spec.image_spec.normalise(meta.at("images").at(image), base)

        converter = Converter(convert=convert_image, convert_path=["images", image])
        configuration.add_converter(converter)

        def convert_tasks(path, val):
            spec = harpoon_spec.tasks_spec(available_actions)
            meta = Meta(path.configuration.root(), []).at("images").at(image).at("tasks")
            configuration.converters.started(path)
            tasks = spec.normalise(meta, val)
            for task in tasks.values():
                task.image = image
            return tasks

        converter = Converter(convert=convert_tasks, convert_path=["images", image, "tasks"])
        configuration.add_converter(converter)