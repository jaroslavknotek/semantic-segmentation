import itertools 
def unzip2(iterator):
    xs, ys = itertools.tee(iterator)
    return (x[0] for x in xs), (y[1] for y in ys)
