import numpy as np
cimport numpy as np

import logging
logger = logging.getLogger("vision.svm")

cdef extern from *:
    ctypedef char* const_char_ptr "const char*"

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)

cdef extern from "liblinear/linear.h":
    cdef struct feature_node:  
        int index
        double value

    cdef struct problem:
        int l # number of training data
        int n # number of features (incl. bias)
        int *y # array of target values
        feature_node **x # array of feature_nodes
        double bias

    cdef struct parameter:
        int solver_type

        # these are for training only
        double eps #stopping criteria 
        double C
        int nr_weight
        int *weight_label
        double* weight

    cdef struct model:
        parameter param
        int nr_class # number of classes 
        int nr_feature
        double *w
        int *label # label of each class (label[n])
        double bias

    char *check_parameter(problem *prob, parameter *param) nogil
    model *liblinear_train "train" (problem *prob, parameter *param) nogil

    cdef enum solver_type:
        L2R_LR,
        L2R_L2LOSS_SVC_DUAL,
        L2R_L2LOSS_SVC,
        L2R_L1LOSS_SVC_DUAL,
        MCSVM_CS,
        L1R_L2LOSS_SVC,
        L1R_LR

MACH_L1R_L2LOSS_SVC = 5
MACH_L1R_LR = 6
MACH_L2R_L1LOSS_SVC_DUAL = 3
MACH_L2R_L2LOSS_SVC = 2
MACH_L2R_L2LOSS_SVC_DUAL = 1
MACH_L2R_LR = 0
MACH_MCSVM_CS = 4

class Model(object):
    """
    A SVM-trained model that stores the weights and bias.
    """
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

def sanity(pos, neg):
    """
    Performs a sanity check on the positive and negative data. Useful for
    debugging training, especially if liblinear is segfaulting.
    """
    logger.info("Performing sanity check on input data")
    if len(pos) == 0:
        raise ValueError("Need at least one positive data point")
    if len(neg) == 0:
        raise ValueError("Need at least one negative data point")
    size = pos[0].size
    for x, point in enumerate(pos + neg):
        if point.ndim != 1:
            raise ValueError("Each data point must be exactly "
            "1 dimension [{0}]".format(x))
        if point.size != size:
            raise ValueError("Each data point must have the "
            "same size [{0}]".format(x))

    logger.debug("Sanity check passed")

cpdef train(positives, negatives, float c = 1.0,
    mach = L2R_L2LOSS_SVC_DUAL, float eps = 0.01,
    float posc = 1.0, float negc = 1.0):
    """
    Trains a linear SVM with positives and negatives.

    positive and negative should be lists of numpy vectors with the
    respective features. Sparse vectors are *not* supported at this time.

    c is the cost of a constraint violation.

    eps is the stopping criterion.

    Returns the learned the weights and bias through a Model object.
    """
    logger.info("Constructing SVM problems and parameters")

    cdef int numpos = len(positives), numneg = len(negatives)
    cdef int numall = numpos + numneg
    cdef int numfeat = positives[0].size
    cdef int i # counter

    logger.debug("Constructing problem")
    cdef problem prob
    prob.l = numall
    prob.n = numfeat + 1
    prob.y = <int*> malloc(numall * sizeof(int))
    for i from 0 <= i < numpos:
        prob.y[i] = 1
    for i from 0 <= i < numneg:
        prob.y[numpos + i] = -1
    prob.bias = 1
    prob.x = <feature_node**> malloc(numall * sizeof(feature_node*))
    for i from 0 <= i < numpos:
        prob.x[i] = build_feature_node(positives[i])
    for i from 0 <= i < numneg:
        prob.x[numpos + i] = build_feature_node(negatives[i])

    logger.debug("Constructing parameter")
    cdef parameter param
    param.solver_type = mach
    param.eps = eps
    param.C = c
    param.nr_weight = 2
    param.weight = <double*> malloc(param.nr_weight*sizeof(double))
    param.weight[0] = posc
    param.weight[1] = negc
    param.weight_label = <int*> malloc(param.nr_weight*sizeof(int))
    param.weight_label[0] = +1
    param.weight_label[1] = -1

    logger.debug("Checking parameters")
    cdef const_char_ptr message = <char*> check_parameter(&prob, &param)
    if message:
        raise RuntimeError("Error training SVM: " + str(message))

    logger.info("Optimizing SVM with liblinear")
    cdef model *mod = liblinear_train(&prob, &param)
    cdef np.ndarray[np.double_t, ndim=1] weights = np.zeros(mod.nr_feature)
    for i from 0 <= i < mod.nr_feature:
        weights[i] = mod.w[i]
    cdef double bias = mod.w[mod.nr_feature]

    logger.debug("Cleanup")
    free(param.weight)
    free(param.weight_label)
    free(prob.y)
    for i from 0 <= i < numall:
        free(prob.x[i])
    free(prob.x)

    return Model(weights, bias)

cdef inline feature_node *build_feature_node(vector):
    """
    Builds a feature node for a vector. Includes a bias term.
    """
    cdef np.ndarray[np.double_t, ndim=1] vectort = vector
    cdef int i, n = vector.size
    cdef feature_node *nodes
    nodes = <feature_node*> malloc((n+2)*sizeof(feature_node))

    for i from 0 <= i < n:
        nodes[i].index = i + 1
        nodes[i].value = vectort[i]
    nodes[n].index = n + 1
    nodes[n].value = 1
    nodes[n + 1].index = -1
    return nodes
