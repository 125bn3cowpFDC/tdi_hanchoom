import numpy as np

# Joint index:
# {0,  "Nose"}  -->  0
# {1,  "Neck"}  -->  17,
# {2,  "RShoulder"}  -->  5,
# {3,  "RElbow"}  -->  7,
# {4,  "RWrist"}  -->  9,
# {5,  "LShoulder"}  -->  6,
# {6,  "LElbow"}  -->  8,
# {7,  "LWrist"}  -->  10,
# {8,  "RHip"}  -->  11,
# {9,  "RKnee"}  -->  13,
# {10, "RAnkle"}  -->  15,
# {11, "LHip"}  -->  12,
# {12, "LKnee"}  -->  14,
# {13, "LAnkle"}  -->  16,
# {14, "REye"}  -->  1,
# {15, "LEye"}  -->  2,
# {16, "REar"}  -->  3,
# {17, "LEar"}  -->  4,

# Edge format: (origin, neighbor)
num_node = 18
self_link = [(i, i) for i in range(num_node)]
#changed by mediapipe
inward = [(9, 7), (7, 5), (10, 8), (8, 6), (16, 14), (14, 12), (15, 13), (13, 11),
          (12, 6), (11, 5), (6, 17), (5, 17), (0, 17), (2, 0), (1, 0), (4, 2),
          (3, 1)]

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

def edge2mat(link, num_node):
    print("link", link)
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    #print(A)
    Dl = np.sum(A, 0)
    #print("ddd",Dl)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD




def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    #print("i",I)
    In = normalize_digraph(edge2mat(inward, num_node))
    #print("in",edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    #print("out",edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A



class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Arguments:
        labeling_mode: must be one of the follow candidates
            uniform: Uniform Labeling
            dastance*: Distance Partitioning*
            dastance: Distance Partitioning
            spatial: Spatial Configuration
            DAD: normalized graph adjacency matrix
            DLD: normalized graph laplacian matrix

    For more information, please refer to the section 'Partition Strategies' in our paper.

    """

    def __init__(self, labeling_mode='uniform'):
        self.A = self.get_adjacency_matrix(labeling_mode) #spatial
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        elif labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)

        else:
            raise ValueError()
        return A


def main():
    mode = ['spatial']
    np.set_printoptions(threshold=1)
    for m in mode:
        print('=' * 10 + m + '=' * 10)
        print(Graph(m).get_adjacency_matrix().size())


if __name__ == '__main__':
    main()
