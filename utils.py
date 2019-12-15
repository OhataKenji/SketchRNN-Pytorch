def ns(x):
    Ns = []
    for i in range(x.shape[1]):
        Ns.append(endindex(x[:, i, :]))

    return Ns


def endindex(x):
    Nmax = x.shape[0]
    for i in range(Nmax):
        if x[i, 4] == 1:
            return i
    return Nmax-1
