from core.softmax import softmax

def simple_test_softmax():

    x = [1.0, 2.0, 3.0]

    probs = softmax(x)

    assert (len(probs)) ==3
    assert abs(sum(probs) - 1.0) <1e-6
    assert probs[2] > probs[1] > probs[0]

def test_soft_stability():

    x = [1000.0, 1001.0, 1002.0]
    probs = softmax(x)

    assert abs(sum(probs) - 1.0) < 1e-6