def count(l,n):
    count = [0 for _ in range(n)]
    for i in range(n):
        count[i] = l.count(i)
    return count
def getCategory(name):
    for i,elm in enumerate(block_annotation[0]):
        if elm ==name :
            return block_annotation[1][i]
    return -1

def one_hot_n(x,n):
    arr = [0 for _ in range(n)]
    arr[x] = 1
    return arr;

def one_hot(x):
    return one_hot_n(x,N_categories)

