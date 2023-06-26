import numpy as np

def BallClustering(distance,epsilon,min_samples = -1):
    labels = np.zeros((distance.shape[0])) - 1
    label_num = 0

    if min_samples == -1 :
    
        for i in range(len(labels)):
            
            if labels[i] == -1 : 
                dist = distance[i,:]
                idx = np.argwhere(dist < epsilon)
                labels[idx] = label_num
                label_num += 1

    else : 

        for i in range(len(labels)):
            
            if labels[i] == -1 : 
                dist = distance[i,:]
                idx = np.argwhere(dist < epsilon)
                if len(idx) >= min_samples : 
                    labels[idx] = label_num
                    label_num += 1

    return labels
        
