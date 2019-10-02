from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def make_A_matrix_small_room(size,side):
    """
    Creates the A matrix for small rooms(left(L) or right(R))
    Parameters:
    size: size of room matrix (scalar)
    side: the side where the Neuman condition is (L/R)
    """
    A_size = (size-1)*(size-2)
    A = np.diag(np.ones(A_size)*-4)
    for i in range(1,A_size):
        if i%(size-1):
            A[i-1][i] = 1
            A[i][i-1] = 1
    for i in range(size-1,A_size):
        A[i-size+1][i] = 1
        A[i][i-size+1] = 1

    #Neumann Conditions
    for i in range(size-2):
        if side == "L":
            bi = (size - 1)*i
            A[bi][bi] = -3
        elif side == "R":
            bi = (size - 2) + (size-1)*i
            A[bi][bi] = -3
    return A

def make_A_matrix_big_room(size):
    """
    Creates the A matrix for the big room
    Parameters:
    size: tuple on the form (m,n)
    """
    m,n = size
    A_size = (m-2)*(n-2)
    A = np.diag(np.ones(A_size)*-4)
    for i in range(1,A_size):
        if i%(n-2):
            A[i-1][i] = 1
            A[i][i-1] = 1
    for i in range(A_size-n+2):
        A[i][i+n-2] = 1
        A[i+n-2][i] = 1
    return A


def make_B_vector(left,top,right,bottom):
    """
    Creates the B vector for a room
    Parameters:
    left: left wall vector
    top: top wall vector
    right: right wall vector
    bottom: bottom wall vector
    4 vectors corresponding to the elements on either side of the matrix.
    If the matrix is of size n x m the top and bottom vectors will be n-2.
    The left and right will be m-2.
    For left and right 0 is the top element.
    """
    if not left.shape == right.shape or not top.shape == bottom.shape:
        print("Error, wrong shapes")

    rows = left.size
    cols = top.size
    length = rows*cols
    b = np.zeros(length)

    # Corner 1
    b[0] = top[0]+left[0]

    # rows - 2 element top
    b[1:cols-1] = top[1:-1]

    # Corner 2
    b[cols-1] = top[-1]+right[0]

    # Mid points
    for i in range(1,rows-1):
        b[i*cols] = left[i]
        b[i*cols+cols-1] = right[i]

    # Corner 3
    b[length-cols] = bottom[0]+left[-1]

    # rows - 2 element bottom
    b[length-cols+1:-1] = bottom[1:-1]

    # Corner 4
    b[length-1] = bottom[-1]+right[-1]
    
    return b

def gamma(u_left, u_right, dx):
    """
    Caluclates the gamma value for two points
    Parameters:
    u_left: the left point
    u_right: the right point
    """
    return (u_right - u_left)/dx

inv_dx = 3
dx = 1/inv_dx
omega = 0.8

#Temperatures for different kind of walls
wallHeaterT = 40
wallNormalT = 15
wallWindowT = 5

is_tb = inv_dx # inner small room side top bot
is_rl = inv_dx - 1 # inner small room side right left
os_side = inv_dx+1 # outer small matrix side

ib_tb = inv_dx-1 # inner big room side top bot
ib_rl = (inv_dx*2)-1 # inner big room side right left
ob_tb = inv_dx+1 # outer matrix big room top bottom
ob_rl = (inv_dx*2)+1 # outer matrix big room right left

#Initiliaze vectors,matrix and values for corresponding room down below

#Middle room

#Temperature vectors for the corresponding wall: left,top,bottom,right(Middle room)
l_mid = np.ones(ib_rl)*wallNormalT
t_mid = np.ones(ib_tb)*wallHeaterT
b_mid = np.ones(ib_tb)*wallWindowT
r_mid = np.ones(ib_rl)*wallNormalT


#Creates the A matrix for the middle room
a_mid = make_A_matrix_big_room((ob_rl,ob_tb))/dx/dx

#Left room

#Temperature vectors for the corresponding wall: left,top,bottom(Left room)
l_small_left = np.ones(is_rl)*wallHeaterT
t_small = np.ones(is_tb)*wallNormalT # Also used in right room
b_small = np.ones(is_tb)*wallNormalT # Also used in right room

#Creates the A matrix for the left room
a_left = make_A_matrix_small_room(os_side, "R")/dx/dx

#Right room

#Temperature vectors for the corresponding wall: right(Right room)
r_small_right = np.ones(is_rl)*wallHeaterT

#Creates the A matrix for the right room, creates the b vector and solves the system
a_right = make_A_matrix_small_room(os_side, "L")/dx/dx


#np.set_printoptions(precision=2) # Uncomment to limit displayed double precision

for iteration in range(4):
    # Solve big room
    b_vector_mid = make_B_vector(l_mid,t_mid,r_mid,b_mid)/dx/dx
    new_x_mid = np.linalg.solve(a_mid,-b_vector_mid)
    if iteration > 0 :
        x_mid = omega*x_mid + (1-omega)*new_x_mid
    else:
        x_mid = new_x_mid
    # Calculate gammas for left small room
    left_border_points = l_mid[-is_rl:]
    left_inner_points = x_mid.reshape((ib_rl,ib_tb))[-is_rl:,0]
    left_gammas = np.zeros(is_rl)
    for i in range(is_rl):
        left_gammas[i] = gamma(left_border_points[i],left_inner_points[i],dx)
        
    # Calculate gammas for right small room
    right_border_points = r_mid[:is_rl]
    right_inner_points = x_mid.reshape((ib_rl,ib_tb))[:is_rl,-1]
    right_gammas = np.zeros(is_rl)
    for i in range(is_rl):
        right_gammas[i] = gamma(right_inner_points[i],right_border_points[i],dx)
        

    # Solve left room
    b_left = make_B_vector(l_small_left,t_small,dx*left_gammas,b_small)/dx/dx
    new_x_left = np.linalg.solve(a_left, -b_left)
    if iteration > 0 :
        x_left = omega*x_left +(1-omega)*new_x_left
    else:
        x_left = new_x_left
    # Update l_mid
    l_mid[-is_rl:] = x_left.reshape((is_rl,is_tb))[:,-1]

    # Solve right room
    b_right = make_B_vector(-dx*right_gammas,t_small,r_small_right,b_small)/dx/dx # minus on gammas acording to formula
    new_x_right = np.linalg.solve(a_right, -b_right)
    if iteration > 0 :
        print(x_right)
        x_right = omega*x_right +(1-omega)*new_x_right
        print(x_right)

    else:
        x_right = new_x_right
    # Update r_mid
    r_mid[:is_rl] = x_right.reshape((is_rl,is_tb))[:,0]

res_left = x_left.reshape((is_rl,is_tb))
res_mid = x_mid.reshape((ib_rl,ib_tb))
res_right = x_right.reshape((is_rl,is_tb))


# Plotting

mid_h,mid_w = res_mid.shape
left_h,left_w = res_left.shape
right_h,right_w = res_right.shape

plot_matrix_width = mid_w+left_w+right_w+2
plot_matrix_height = mid_h+2
plot_matrix = np.zeros((plot_matrix_height,plot_matrix_width))


plot_matrix[-inv_dx:-1,1:inv_dx+1] = res_left
plot_matrix[1:-1,inv_dx+1:2*inv_dx] = res_mid
plot_matrix[1:inv_dx,2*inv_dx:-1] = res_right
plot_matrix[-1,1:inv_dx+1] = np.ones(inv_dx)*15
plot_matrix[-1,inv_dx+1:2*inv_dx] = np.ones(inv_dx-1)*5
plot_matrix[-inv_dx-1:,0] = np.ones(inv_dx+1)*40
plot_matrix[-inv_dx-1,1:inv_dx+1] = np.ones(inv_dx)*15
plot_matrix[:inv_dx,inv_dx] = np.ones(inv_dx)*15
plot_matrix[0,inv_dx+1:2*inv_dx] = np.ones(mid_w)*40
plot_matrix[:inv_dx+1,-1] = np.ones(inv_dx+1)*40
plot_matrix[inv_dx+1:,2*inv_dx] = np.ones(inv_dx)*15
plot_matrix[inv_dx,2*inv_dx:-1] = np.ones(inv_dx)*15
plot_matrix[0,2*inv_dx:-1] = np.ones(right_w)*15

heatplot = plt.imshow(plot_matrix)
heatplot.set_cmap('hot')
plt.colorbar()
plt.show()