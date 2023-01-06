import numpy as np

def reciprocal_vec(model):
    #returns reciprocal lattice vectors
    lat = model.get_lat()     
    a_matrix = np.array([[lat[1,1], -lat[0,1]],[-lat[1,0], lat[0,0]]])
    b_matrix = (2.*np.pi / (lat[0,0]*lat[1,1]-lat[0,1]*lat[1,0])) * a_matrix

    b1 = b_matrix[:,0].reshape(-1,1)
    b2 = b_matrix[:,1].reshape(-1,1)
    return b1, b2

def orb_cart (model):
    #returns position of orbitals in cartesian coordinates
    n_orb = model.get_num_orbitals()
    lat_super = model.get_lat()          
    orb_red = model.get_orb()            

    orb_c = []
    for i in range (n_orb):
        orb_c.append( (np.matmul(lat_super.transpose(),orb_red[i].reshape(-1,1))).squeeze() )   
    orb_c = np.array(orb_c)
    return orb_c

def orb_cart_spin (model):
    #returns position of orbitals in cartesian coordinates
    n_occ = model.get_num_orbitals()
    lat_super = model.get_lat()          
    orb_red = model.get_orb()            

    orb_c = []
    for i in range (n_occ):
        orb_c.append( (np.matmul(lat_super.transpose(),orb_red[i].reshape(-1,1))).squeeze() )  
        orb_c.append( (np.matmul(lat_super.transpose(),orb_red[i].reshape(-1,1))).squeeze() )  
    orb_c = np.array(orb_c)
    return orb_c

def periodic_gauge (u_n0, b, model):
    #returns the matrix of occupied eigenvectors at the edge of the Brillouin zone imposing periodic gauge
    n_orb = model.get_num_orbitals()
    n_occ = n_orb//2

    orb_c = orb_cart(model)
    vec_scal_b = orb_c @ b
    vec_exp_b = np.exp(-1.j*vec_scal_b)

    u_nb = vec_exp_b.T * u_n0[:n_occ,:]

    return u_nb

def periodic_gauge_spin(u_n0, b, model):
    #returns the matrix of occupied eigenvectors at the edge of the Brillouin zone imposing periodic gauge
    n_orb = model.get_num_orbitals()
    n_occ = n_orb

    orb_c = orb_cart_spin(model)
    vec_scal_b = orb_c @ b
    vec_exp_b = np.exp(-1.j*vec_scal_b)

    u_nb = vec_exp_b.T * u_n0[:n_occ,:]

    return u_nb



