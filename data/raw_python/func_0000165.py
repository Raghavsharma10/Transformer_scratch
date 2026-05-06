def write_cert_items(xml_tree, records, api_ver=3, app_id=None, app_ver=None):
    """Generate the certificate blocklists.

    <certItem issuerName="MIGQMQswCQYD...IENB">
      <serialNumber>UoRGnb96CUDTxIqVry6LBg==</serialNumber>
    </certItem>

    or

    <certItem subject='MCIxIDAeBgNVBAMMF0Fub3RoZXIgVGVzdCBFbmQtZW50aXR5'
              pubKeyHash='VCIlmPM9NkgFQtrs4Oa5TeFcDu6MWRTKSNdePEhOgD8='>
    </certItem>
    """
    if not records or not should_include_certs(app_id, app_ver):
        return

    certItems = etree.SubElement(xml_tree, 'certItems')
    for item in records:
        if item.get('subject') and item.get('pubKeyHash'):
            cert = etree.SubElement(certItems, 'certItem',
                                    subject=item['subject'],
                                    pubKeyHash=item['pubKeyHash'])
        else:
            cert = etree.SubElement(certItems, 'certItem',
                                    issuerName=item['issuerName'])
            serialNumber = etree.SubElement(cert, 'serialNumber')
            serialNumber.text = item['serialNumber']