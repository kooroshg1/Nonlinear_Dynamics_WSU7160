__author__ = 'koorosh'
import matplotlib.pylab as plt
import numpy as np

def wedge(theta = 0, thetaDot = 0):
    # Define vertices coordiates
    vertCoord = np.array([[-5,0],
                          [0,1],
                          [5, 0],
                          [0, -1]])
    # # number of node between each two vertices (include edge vertices)
    # N = 3
    # n = N + 2
    # nodeCoord = np.zeros([4*N + 4, 2])
    #
    # nodeCoord[0, :] = vertCoord[0, :]
    # nodeCoordX = np.linspace(vertCoord[0, 0], vertCoord[1, 0], n)
    # nodeCoordY = (vertCoord[1, 1] - vertCoord[0, 1]) / \
    #              (vertCoord[1, 0] - vertCoord[0, 0]) * nodeCoordX + vertCoord[1, 1]
    # nodeCoord[1:N+1, 0] = nodeCoordX[1:-1]
    # nodeCoord[1:N+1, 1] = nodeCoordY[1:-1]
    #
    # nodeCoord[N+1, :] = vertCoord[1, :]
    # nodeCoordX = np.linspace(vertCoord[1, 0], vertCoord[2, 0], n)
    # nodeCoordY = (vertCoord[2, 1] - vertCoord[1, 1]) / \
    #              (vertCoord[2, 0] - vertCoord[1, 0]) * nodeCoordX + vertCoord[1, 1]
    # nodeCoord[N+2:2*N+2, 0] = nodeCoordX[1:-1]
    # nodeCoord[N+2:2*N+2, 1] = nodeCoordY[1:-1]
    #
    # nodeCoord[2*N+2, :] = vertCoord[2, :]
    # nodeCoordX = np.linspace(vertCoord[2, 0], vertCoord[3, 0], n)
    # nodeCoordY = (vertCoord[3, 1] - vertCoord[2, 1]) / \
    #              (vertCoord[3, 0] - vertCoord[2, 0]) * nodeCoordX + vertCoord[3, 1]
    # nodeCoord[2*N+3:3*N+3, 0] = nodeCoordX[1:-1]
    # nodeCoord[2*N+3:3*N+3, 1] = nodeCoordY[1:-1]
    #
    # nodeCoord[3*N+3, :] = vertCoord[3, :]
    # nodeCoordX = np.linspace(vertCoord[3, 0], vertCoord[0, 0], n)
    # nodeCoordY = (vertCoord[0, 1] - vertCoord[3, 1]) / \
    #              (vertCoord[0, 0] - vertCoord[3, 0]) * nodeCoordX + vertCoord[3, 1]
    # nodeCoord[3*N+4:4*N+4, 0] = nodeCoordX[1:-1]
    # nodeCoord[3*N+4:4*N+4, 1] = nodeCoordY[1:-1]
    #
    # np.savetxt('coord.txt', nodeCoord, '%2.2f')

    # number of node between each two vertices (exclude edge vertices)
    N = 3
    n = N + 2
    nodeCoord = np.zeros([4*N, 2])
    nodeGrad = np.zeros([4*N, 2])
    nodeTangent = np.zeros([4*N, 2])
    nodeNormal = np.zeros([4*N, 2])
    # -------------------------------------------------------------------------
    nodeCoordX = np.linspace(vertCoord[0, 0], vertCoord[1, 0], n)
    nodeCoordY = (vertCoord[1, 1] - vertCoord[0, 1]) / \
                 (vertCoord[1, 0] - vertCoord[0, 0]) * nodeCoordX + vertCoord[1, 1]
    nodeCoord[:N, 0] = nodeCoordX[1:-1]
    nodeCoord[:N, 1] = nodeCoordY[1:-1]
    nodeGrad[:N, 0] = nodeCoordX[1:-1]
    nodeGrad[:N, 1] = (vertCoord[1, 1] - vertCoord[0, 1]) / \
                      (vertCoord[1, 0] - vertCoord[0, 0]) * np.ones(N)
    nodeTangent[:N, 0] = vertCoord[1, 0] - vertCoord[0, 0]
    nodeTangent[:N, 1] = vertCoord[1, 1] - vertCoord[0, 1]
    # -------------------------------------------------------------------------
    nodeCoordX = np.linspace(vertCoord[1, 0], vertCoord[2, 0], n)
    nodeCoordY = (vertCoord[2, 1] - vertCoord[1, 1]) / \
                 (vertCoord[2, 0] - vertCoord[1, 0]) * nodeCoordX + vertCoord[1, 1]
    nodeCoord[N:2*N, 0] = nodeCoordX[1:-1]
    nodeCoord[N:2*N, 1] = nodeCoordY[1:-1]
    nodeGrad[N:2*N, 0] = nodeCoordX[1:-1]
    nodeGrad[N:2*N, 1] = (vertCoord[2, 1] - vertCoord[1, 1]) / \
                         (vertCoord[2, 0] - vertCoord[1, 0]) * np.ones(N)
    nodeTangent[N:2*N, 0] = vertCoord[2, 0] - vertCoord[1, 0]
    nodeTangent[N:2*N, 1] = vertCoord[2, 1] - vertCoord[1, 1]
    # -------------------------------------------------------------------------
    nodeCoordX = np.linspace(vertCoord[2, 0], vertCoord[3, 0], n)
    nodeCoordY = (vertCoord[3, 1] - vertCoord[2, 1]) / \
                 (vertCoord[3, 0] - vertCoord[2, 0]) * nodeCoordX + vertCoord[3, 1]
    nodeCoord[2*N:3*N, 0] = nodeCoordX[1:-1]
    nodeCoord[2*N:3*N, 1] = nodeCoordY[1:-1]
    nodeGrad[2*N:3*N, 0] = nodeCoordX[1:-1]
    nodeGrad[2*N:3*N, 1] = (vertCoord[3, 1] - vertCoord[2, 1]) / \
                           (vertCoord[3, 0] - vertCoord[2, 0]) * np.ones(N)
    nodeTangent[2*N:3*N, 0] = vertCoord[3, 0] - vertCoord[2, 0]
    nodeTangent[2*N:3*N, 1] = vertCoord[3, 1] - vertCoord[2, 1]
    # -------------------------------------------------------------------------
    nodeCoordX = np.linspace(vertCoord[3, 0], vertCoord[0, 0], n)
    nodeCoordY = (vertCoord[0, 1] - vertCoord[3, 1]) / \
                 (vertCoord[0, 0] - vertCoord[3, 0]) * nodeCoordX + vertCoord[3, 1]
    nodeCoord[3*N:4*N, 0] = nodeCoordX[1:-1]
    nodeCoord[3*N:4*N, 1] = nodeCoordY[1:-1]
    nodeGrad[3*N:4*N, 0] = nodeCoordX[1:-1]
    nodeGrad[3*N:4*N, 1] = (vertCoord[0, 1] - vertCoord[3, 1]) / \
                           (vertCoord[0, 0] - vertCoord[3, 0]) * np.ones(N)
    nodeTangent[3*N:4*N, 0] = vertCoord[0, 0] - vertCoord[3, 0]
    nodeTangent[3*N:4*N, 1] = vertCoord[0, 1] - vertCoord[3, 1]

    # Define rotation in clockwise with respect to [0,0]
    # theta = 0
    # thetaDot = 0
    theta = theta * np.pi / 180
    thetaDot = thetaDot * np.pi / 180
    rMat = np.matrix([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    rMatDot = thetaDot * np.matrix([[-np.sin(theta), -np.cos(theta)],
                                    [np.cos(theta), -np.sin(theta)]])
    rMat90 = np.matrix([[np.cos(np.pi/2), -np.sin(np.pi/2)],
                        [np.sin(np.pi/2), np.cos(np.pi/2)]])

    nodeCoordDot = np.zeros([4*N, 2])
    for ni in range(0, nodeCoord.shape[0]):
        nodeCoord[ni, :] = (rMat * nodeCoord[ni, :].reshape([2, 1])).reshape([1, 2])
        nodeCoordDot[ni, :] = (rMatDot * nodeCoord[ni, :].reshape([2, 1])).reshape([1, 2])
        nodeGrad[ni, 1] = (nodeGrad[ni, 1] + np.tan(theta)) / (1 - nodeGrad[ni, 1] * np.tan(theta))
        nodeTangent[ni, :] = nodeTangent[ni, :] / np.linalg.norm(nodeTangent[ni, :])
        nodeTangent[ni, :] = (rMat * nodeTangent[ni, :].reshape([2, 1])).reshape([1, 2])
        nodeNormal[ni, :] = (rMat90 * nodeTangent[ni, :].reshape([2, 1])).reshape([1, 2])


    np.savetxt('coord.txt', nodeCoord, '%2.2f')
    np.savetxt('coordDot.txt', nodeCoordDot, '%2.2f')
    np.savetxt('grad.txt', nodeGrad, '%2.2f')
    # np.savetxt('tangent.txt', nodeTangent, '%2.2f')
    np.savetxt('normal.txt', nodeNormal, '%2.2f')

    # plt.figure()
    # plt.plot(nodeCoord[:, 0], nodeCoord[:, 1], 'ko')
    # plt.axis('equal')
    # plt.show()
