import numpy as np

def l1_distance(a, b):
    """Computes the Euclidean distance matrix between a and b.

    Inputs:
        A: D x M array.
        B: D x N array.

    Returns:
        E: M x N Euclidean distances between vectors in A and B.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B should be of same dimensionality")

    #리턴할 배열의 틀을 만듭니다.
    back = np.zeros([a.shape[1],b.shape[1]])

    # 벡터갯수 x 성분 이 되도록 벡터를 변환해줍니다.
    a = a.T
    b = b.T

    #한줄씩 거리를 구하여 추가합니다.
    for i in range(a.shape[0]):
        back[i] = np.sum(np.abs(b - a[i]),axis = 1)

    return back


def l2_distance(a, b):
    """Computes the Euclidean distance matrix between a and b.

    Inputs:
        A: D x M array.
        B: D x N array.

    Returns:
        E: M x N Euclidean distances between vectors in A and B.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B should be of same dimensionality")

    aa = np.sum(a**2, axis=0)
    bb = np.sum(b**2, axis=0)
    ab = np.dot(a.T, b)

    return np.sqrt(aa[:, np.newaxis] + bb[np.newaxis, :] - 2*ab)

if __name__ == "__main__":
    import utils
    train_img, train_label = utils.load_train()
    val_img, val_label  = utils.load_valid()
    print(l2_distance(train_img.T,val_img.T))

