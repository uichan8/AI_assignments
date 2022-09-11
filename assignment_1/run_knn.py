import numpy as np
from l2_distance import l2_distance, l1_distance

def run_knn(k, train_data, train_labels, valid_data ,distance = "l2"):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples, 
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification 
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data 
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels 
                      for the validation data.
    """

    # call distance to compute distance between valid data and train data
    if distance == "l1":
        dis_array = l1_distance(train_data,valid_data)
    elif distance == "l2":
        dis_array = l2_distance(train_data,valid_data)
    else:
        return

    # sort the distance to get top k nearest data indices
    arg = np.argsort(dis_array,axis = 0)
    nearest = arg[:k]
    
    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # note this only works for binary labels
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1,1)

    return nearest.shape

if __name__ == "__main__":
    import utils
    train_img, train_label = utils.load_train()
    val_img, val_label  = utils.load_valid()
    print(run_knn(3,train_img.T,train_label,val_img.T))
    print(val_img.shape)
    print(train_img.shape)

    