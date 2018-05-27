import numpy as np

def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6

def R_to_angle(Rt):
# Ground truth pose is present as [R | t] 
# R: Rotation Matrix, t: translation vector
# transform matrix to angles
	Rt = np.reshape(np.array(Rt), (3,4))
	t = Rt[:,-1]
	R = Rt[:,:3]

	assert(isRotationMatrix(R))
	
	sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0]) 
	singular = sy < 1e-6
 
	if not singular :
		x = np.arctan2(R[2,1] , R[2,2])
		y = np.arctan2(-R[2,0], sy)
		z = np.arctan2(R[1,0], R[0,0])
	else :
		x = np.arctan2(-R[1,2], R[1,1])
		y = np.arctan2(-R[2,0], sy)
		z = 0
	theta = [x, y, z]
	pose_6 = np.concatenate((theta, t))
	assert(pose_6.shape == (6,))
	return pose_6

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R