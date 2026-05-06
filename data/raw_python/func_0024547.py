def set_xml(self, diagram, force=False):
        """
        updates xml link if there aren't any running instances of this wf
        Args:
            diagram: XMLDiagram object
        """
        no_of_running = WFInstance.objects.filter(wf=self, finished=False, started=True).count()
        if no_of_running and not force:
            raise RunningInstancesExist(
                "Can't update WF diagram! Running %s WF instances exists for %s" % (
                    no_of_running, self.name
                ))
        else:
            self.xml = diagram
            parser = BPMNParser(diagram.body)
            self.description = parser.get_description()
            self.title = parser.get_name() or self.name.replace('_', ' ').title()
            extensions = dict(parser.get_wf_extensions())
            self.programmable = extensions.get('programmable', False)
            self.task_type = extensions.get('task_type', None)
            self.menu_category = extensions.get('menu_category', settings.DEFAULT_WF_CATEGORY_NAME)
            self.save()