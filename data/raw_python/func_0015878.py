def add(self, service_id, request_id, description=None, details=None):
        if not service_id:
            raise ValueError('service_id is required')

        if not request_id:
            raise ValueError('request_id is required')

        """
        curl -X POST \
            -H 'x-ca-version: 1.0' \
            -H 'x-ca-accesskeyid: YourAccessId' \
            -d "service_id=p4dkh2sg&request_id=c13ed5aa-d6d2-11e8-ba11-02420a582a05&description=blahlblah" \
            https://api.productai.cn/bad_cases/_0000204
        """
        data = dict()
        data['service_id'] = service_id
        data['request_id'] = request_id

        if description:
            data['description'] = description

        if details:
            data['details'] = details

        """
{
"created_at": "2018-10-24T03:30:51Z",
"description": "\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5",
"details": "",
"id": 34,
"image_path": null,
"modified_at": "2018-10-24T03:30:51Z",
"reporter_id": 1632,
"request_id": "34954696-d73d-11e8-9419-0242ac1c2b04",
"service_id": "p4dkh2sg",
"status": "open"
}            
        """
        return self.client.post(self.base_url, data=data)