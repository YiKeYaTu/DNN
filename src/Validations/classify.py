def classify(y, _y): 
    i = 0
    _i = 0

    for j in range(len(y)):
        if y[j] > y[i]:
            i = j
        
        if _y[j] > y[_i]:
            _i = j

    return i == _i