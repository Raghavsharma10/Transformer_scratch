def parse_workflow_messages(self):
        """
        Transmits client message that defined in
        a workflow task's inputOutput extension

       .. code-block:: xml

            <bpmn2:extensionElements>
            <camunda:inputOutput>
            <camunda:inputParameter name="client_message">
            <camunda:map>
              <camunda:entry key="title">Teşekkürler</camunda:entry>
              <camunda:entry key="body">İşlem Başarılı</camunda:entry>
              <camunda:entry key="type">info</camunda:entry>
            </camunda:map>
            </camunda:inputParameter>
            </camunda:inputOutput>
            </bpmn2:extensionElements>

        """
        if 'client_message' in self.current.spec.data:
            m = self.current.spec.data['client_message']
            self.current.msg_box(title=m.get('title'),
                                 msg=m.get('body'),
                                 typ=m.get('type', 'info'))