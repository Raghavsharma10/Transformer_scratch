def Inference(probability=None, relation=None, name=None, concept=None):
    """Represents a probable cause / relation between this event and some prior.

    Args:
        probability(float): Value 0.0 to 1.0.
        relation(str): e.g. 'associated' or 'identified' (see Voevent spec)
        name(str): e.g. name of identified progenitor.
        concept(str): One of a 'formal UCD-like vocabulary of astronomical
            concepts', e.g. http://ivoat.ivoa.net/stars.supernova.Ia - see
            VOEvent spec.
    """
    atts = {}
    if probability is not None:
        atts['probability'] = str(probability)
    if relation is not None:
        atts['relation'] = relation
    inf = objectify.Element('Inference', attrib=atts)
    if name is not None:
        inf.Name = name
    if concept is not None:
        inf.Concept = concept
    return inf