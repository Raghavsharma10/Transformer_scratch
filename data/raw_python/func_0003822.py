def compute_similarity(a, b, margin=1.0, cutoff=10.0):
    """Compute the similarity between two molecules based on their descriptors

       Arguments:
         a  --  the similarity measure of the first molecule
         b  --  the similarity measure of the second molecule
         margin  --  the sensitivity when comparing distances (default = 1.0)
         cutoff  --  don't compare distances longer than the cutoff (default = 10.0 au)

       When comparing two distances (always between two atom pairs with
       identical labels), the folowing formula is used:

       dav = (distance1+distance2)/2
       delta = abs(distance1-distance2)

       When the delta is within the margin and dav is below the cutoff:

         (1-dav/cutoff)*(cos(delta/margin/np.pi)+1)/2

       and zero otherwise. The returned value is the sum of such terms over all
       distance pairs with matching atom types. When comparing similarities it
       might be useful to normalize them in some way, e.g.

       similarity(a, b)/(similarity(a, a)*similarity(b, b))**0.5
    """
    return similarity_measure(
        a.table_labels, a.table_distances,
        b.table_labels, b.table_distances,
        margin, cutoff
    )