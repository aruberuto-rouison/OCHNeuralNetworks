import numpy as np
import math
import matplotlib as mpl
from matplotlib.patches import Arrow
import matplotlib.pyplot as plt
import scipy.optimize as sciopt 
import copy
from IPython.display import display, Math, Markdown
    
plt.rcParams['text.usetex'] = True


def gg(s,epsilon) : return (1./np.pi)*np.arctan(s/epsilon) + 0.5 #  
def g0(s) : 
    if s<=0. : return 0. 
    else : return 1.
def npn(v) :
    return np.linalg.norm(v)

def costFunctional(A, xiflat, u0=np.array([1.,1.]), lamb=1., alph=1., bet=1., nInst=20, ggg=g0) : 
    uu = np.zeros((nInst,2))  
    #print("uu: ",uu)
    xi = [np.array([[0.,xiflat[2*ii]], [xiflat[2*ii+1],0.]]) for ii in range(nInst)]
    #print("xi[0]: ",xi[0])
    uu[0] = (A + xi[0]) @ np.array([ggg(u0[0]), ggg(u0[1])]) 
    #print("uu[0]: ",uu[0])
    for ii in range(1,nInst) :
        uu[ii] = (A + xi[ii]) @ np.array([ggg(uu[ii - 1, 0]), ggg(uu[ii - 1, 1])]) 
#
    cc = np.exp(-lamb*0.) * alph * (npn(uu[0]-u0))**2. + \
            np.sum([np.exp(-lamb*kk) * alph * (npn(uu[kk+1]-uu[kk])**2.) for kk in range(nInst-1)]) + \
            np.sum([np.exp(-lamb*kk) * bet * (npn(xi[kk])**2.) for kk in range(nInst)])
    return cc

def uu_plain_evolution(A, nInst=50, u0=np.array([1., 1.]), epsilon=0.) : 
    #print(" n instants: ",nInst)
    if epsilon==0. : ggg = g0
    else : 
        #def ggg(s) : return (1./np.pi)*np.arctan(s/epsilon) + 0.5
        def ggg(s) : return gg(s,epsilon)
    uu = np.zeros((nInst+1,2))  
    uu[0] = copy.copy(u0) 
    for ii in range(1,nInst+1) :
        uu[ii] = (A) @ np.array([ggg(uu[ii - 1, 0]), ggg(uu[ii - 1, 1])]) 
    # convergence check
    convsteps = math.inf
    for iv,vect in enumerate(zip(uu[:-1], uu[1:])) : 
        xy0, xy1 = vect
        if np.linalg.norm(xy1-xy0)==0. : 
            convsteps = iv
            break
    if math.isinf(convsteps): print('Hop plain evolution not converged... uu=',list(uu[:5]))
    else : 
        print("Hop plain evolution converged to ",uu[-1]," in ",convsteps," steps ... uu=",list(uu[:convsteps+1]))
    return uu

def uu_evo_w_xi(A, xiflat=np.zeros(50), u0=np.array([1., 1.]), epsilon=0.) : 
    nInst=int(len(xiflat))//2
    if epsilon==0. : ggg = g0
    else : 
        def ggg(s) : return (1./np.pi)*np.arctan(s/epsilon) + 0.5
    print("len xiflat: ",len(xiflat)," n instants: ",nInst)
    uu = np.zeros((nInst+1,2))  
    xi = [np.array([[0.,xiflat[2*ii]], [xiflat[2*ii+1],0.]]) for ii in range(nInst)]
    uu[0] = copy.copy(u0) 
    for ii in range(1,nInst+1) :
        uu[ii] = (A + xi[ii-1]) @ np.array([ggg(uu[ii - 1, 0]), ggg(uu[ii - 1, 1])]) 
    return uu

def optim_evolution(A, xiflat, u0=np.array([1., 1.]), nInst=50, lamb=1.0, alph=1.0, bet=1.0, epsilon=0.) :
    # nInstants = 50
    # xxif = np.ones(2*nInstants)
    # u0=np.array([-1.,-3.])
    # print("u0: ",u0)
    bds = [(-0.5,0.5) for _ in range(len(xiflat))]
    if epsilon==0. : ggg = g0
    else : 
        def ggg(s) : return (1./np.pi)*np.arctan(s/epsilon) + 0.5
    
    def costFunct(xif) :
        return costFunctional(A, xif, u0=u0, lamb=lamb,  alph=alph, bet=bet, nInst=nInst, ggg=ggg)
    #res = sciopt.minimize(costFunct,xiflat, method='Nelder-Mead', bounds=bds)
    res = sciopt.minimize(costFunct,xiflat, bounds=bds)
    xifin = res.x
    fval = res.fun
    print("final xi: [",*(res.x[:7]),"...]\nfinal function: ",res.fun)
    #print(r'distance betw xi-start and xi-final: ',np.linalg.norm(xiflat-xifin))
    display(Markdown(rf'distance betw $\xi$-start and $\xi$-final: '+str(np.linalg.norm(xiflat-xifin))))
    uu_evo = uu_evo_w_xi(A, xiflat=xifin, u0=u0, epsilon=epsilon)
    print("evoluzione u[t] sotto xi[t] final, u1= [",*(np.transpose(uu_evo)[0][:7]),"...] e u2= [",*(np.transpose(uu_evo)[1][:7]),r'...] con $\alpha=$',np.round(alph,1),r' e $\beta=$',np.round(bet,1),"\n" )
    # 
    display(Markdown(r'final $u$: '+str(uu_evo[-1])))
    #plt.plot(list(range(len(xifin))),xifin)
    #plt.show()
    return xifin, fval, uu_evo
    #

    
#
#                 PLOT OF THE RESULTS      
#

# Plot of the Vector Field X (quiver)
#
def plotVectorFieldXYUV(A,xi=np.array([[0.,0.],[0.,0.]]),xbds=[-2.,2.5],ybds=[-2.,2.5],step=0.5, ggg=g0) : 
    def XXvec1(x,y,xi=np.array([[0.,0.],[0.,0.]])) :
        return ((A + xi) @ np.array([ggg(x), ggg(y)]))[0]-x
    def XXvec2(x,y,xi=np.array([[0.,0.],[0.,0.]])) :
        return ((A + xi) @ np.array([ggg(x), ggg(y)]))[1]-y
    xx = np.arange(xbds[0], xbds[1], step)
    yy = np.arange(ybds[0], ybds[1], step)
    X, Y = np.meshgrid(xx, yy)
    U = [[XXvec1(X[irx,ix],Y[irx,ix],xi=xi) for ix in range(len(xrow))] for irx,xrow in enumerate(X)]
    V = [[XXvec2(X[irx,ix],Y[irx,ix],xi=xi) for ix in range(len(xrow))] for irx,xrow in enumerate(X)]
    return X, Y, U, V
#
def plotVectorField(A,xi=np.array([[0.,0.],[0.,0.]]),xbds=[-2.,2.5],ybds=[-2.,2.5],step=0.5,ggg=g0) : 
    #
    fig, ax = plt.subplots()
    #
    X,Y,U,V = plotVectorFieldXYUV(A, xi=xi, xbds=xbds, ybds=ybds, step=step, ggg=ggg)
    #
    q = ax.quiver(X, Y, U, V)
    ax.quiverkey(q, X=0.3, Y=1.1, U=10,
             label='Quiver key, length = 10', labelpos='E')
    plt.show()

    

def plot_track(verts, ax, cmap=mpl.colormaps['seismic'],npts = 200, verb=False, **kw_args):
    '''Plot followed track: verts is 2D array: x, y'''
    stones = [verts[0]]
    #
    color_pts = cmap(np.linspace(0,1,npts))
    totarlen = 0.
    n_arcs = len(verts)-1
    arclens, cumarclens = [],[0.]
    # loop check
    continua_a_cercare = True
    loop_found = False
    last_i = len(verts)-1
    for end_i in range(1,len(verts)) : 
        for start_i in range(0,end_i) :
            if npn(verts[end_i] - verts[start_i])==0. : 
                if end_i-start_i == 1 : print("EQUILIBRIUM REACHED:",start_i," == ",end_i)
                else : 
                    loop_found = True
                    if verb : print("LOOP FOUND: ",start_i," == ",end_i)
                    loop_idxs = list(range(start_i,end_i))
                last_i = end_i - 1 
                continua_a_cercare = False
                break
        if not(continua_a_cercare) : break
    for ic, vert in enumerate(zip(verts[:last_i], verts[1:last_i+1])):
        xy0, xy1 = vert
        arlen = npn(xy1 - xy0)
        if arlen == 0. : 
            print("stones terminated at step : ",ic," stones found: ", stones)
            break
        stones.append(xy1)
        arclens.append(arlen)
        totarlen += arlen
        cumarclens.append(totarlen)
    #
    arcstep = totarlen / (npts-1)
    pos = 0
    pts = []
    pts.append(stones[0])
    #
    for ipt in range(1,npts-1) :
        arclen = ipt*arcstep
        while arclen>cumarclens[pos] : pos += 1
        #
        if (cumarclens[pos]-cumarclens[pos-1])==0. :
            print("ERROR: cumarclens identical! : ", (cumarclens[pos]-cumarclens[pos-1]))
            print("len arclens: ",len(arclens)," len cumarclens: ",len(cumarclens))
            break
        eta = (arclen - cumarclens[pos-1])/(cumarclens[pos]-cumarclens[pos-1])
        eta = min(eta,1.)
        pt = stones[pos-1] + eta*(stones[pos]-stones[pos-1])
        pts.append(pt)
    pts.append(stones[-1])
    # points
    ax.scatter(verts[0][0],verts[0][1],s=120,marker='s',ec='b',fc='none')
    # trajectories
    [ax.plot(np.transpose(pts[ii-1:ii+1])[0],np.transpose(pts[ii-1:ii+1])[1],lw=2.,c=color_pts[ii],alpha=0.25, zorder=1) for ii in range(1,len(pts))]
    # targets
    attractors = []
    # one attractor: point - limit exists (plot star) || multiple attractors: cyclic solutions
    if loop_found : 
        attractors = [verts[llid] for llid in loop_idxs]
        # ax.scatter([verts[llid][0] for llid in loop_idxs],[verts[llid][1] for llid in loop_idxs],s=250,marker=(5, 0, 0),fc='g',ec='k')
        # attractors.append(attrac)
    else : 
        # ax.scatter(verts[-1][0],verts[-1][1],s=250,marker='*',fc='r',ec='k')
        attractors = [stones[-1]]
    if verb: print("ATTRACTORS: ",attractors)
        
    return attractors
    



def plotTrajectory(A,uu_evo,xifin,cmap=mpl.colormaps['seismic'],titles=None,figname=None) :
    xmax = max([np.abs(u[0]) for u in uu_evo])
    ymax = max([np.abs(u[1]) for u in uu_evo])
    bd = max([xmax,ymax,1.0])
    fig, axs = plt.subplots(1,3,figsize=(15,4))
    if titles != None : 
        axs[0].title.set_text(r'$u_{1,2}$ evolution ')
        axs[1].title.set_text(r'trajectory + $X_A$ ')
        axs[2].title.set_text(r'trajectory + $X_{A+\xi_f}$ ')
        fig.suptitle(titles)
    axs[0].plot(list(range(len(uu_evo))),np.transpose(uu_evo)[0])
    axs[0].plot(list(range(len(uu_evo))),np.transpose(uu_evo)[1])
    #
    X, Y, U, V = plotVectorFieldXYUV(A,xbds=[(-1)*bd,bd],ybds=[(-1)*bd,bd],step=(bd/10))
    # cseq_track = cmap(np.linspace(0, 1, len(uu_evo)-1))
    # cseq_dots = cmap(np.linspace(0, 1, len(uu_evo)))
    plot_track(uu_evo , axs[1], fill=True, width=0.25, alpha=0.75)
    axs[1].quiver(X, Y, U, V)
    #
    X, Y, U, V = plotVectorFieldXYUV(A+xifin,xbds=[(-1)*bd,bd],ybds=[(-1)*bd,bd],step=(bd/10))
    # cseq_track = cmap(np.linspace(0, 1, len(uu_evo)-1))
    # cseq_dots = cmap(np.linspace(0, 1, len(uu_evo)))
    plot_track(uu_evo , axs[2], fill=True, width=.25, alpha=1.)
    axs[2].quiver(X, Y, U, V)
    if figname!=None : 
        print("figname: ",figname)
        plt.savefig(figname)
    plt.show()

def plot_multiple_tracks(A, tracks,figname=None,title=None,epsilon=0.0,verbose=False,**kwargs) : 
    if epsilon==0. : ggg = g0
    else : 
        #def ggg(s) : return (1./np.pi)*np.arctan(s/epsilon) + 0.5
        def ggg(s) : return gg(s,epsilon)
    cmap = mpl.colormaps['seismic']
    color_verts = cmap(np.linspace(0, 1, len(tracks[0])))
    fig, ax = plt.subplots(figsize=(7,5))
    if title != None: 
        fig.suptitle(title)
    #
    ax.set_xlabel(r'$u_1$', fontsize = 20)
    ax.set_ylabel(r'$u_2$', fontsize = 20)
    bd = max([np.abs(t)*1.2 for ttt in tracks for tt in ttt for t in tt])
    if verbose: print(" boundary found: ",bd," now vector field...")
    #
    X, Y, U, V = plotVectorFieldXYUV(A,xbds=[(-1)*bd,bd],ybds=[(-1)*bd,bd],step=(bd/10), ggg=ggg)
    #
    #
    attracts_all = []
    for trk in tracks :
        attracts_track = plot_track(trk, ax, fill=True, width=0.15, npts=200, alpha=0.75,**kwargs)
        attracts_all.append(attracts_track)
    ax.quiver(X, Y, U, V)
    for attrs in attracts_all :
        if len(attrs)==0 :
            if verbose: print("no attractors: not converged?")
            #ax.scatter(attrs[0][0],attrs[0][1],s=200,marker='D',fc=mpl.colors.to_rgba('b', 0.0),
            #           ec=mpl.colors.to_rgba('r', 0.75), zorder=2)
        elif len(attrs)==1 :
            ax.scatter(attrs[0][0],attrs[0][1],s=200,marker='*',fc=mpl.colors.to_rgba('b', 0.0),
                       ec=mpl.colors.to_rgba('r', 0.75), zorder=2)
        else :
            if verbose: print("sequence of attractors: ",attrs)
            ax.scatter([att[0] for att in attrs],[att[1] for att in attrs],s=200,marker='o',
                            fc=mpl.colors.to_rgba('b', 0.0),ec=mpl.colors.to_rgba('r', 0.75),zorder=2,alpha=0.85)
    #
    if figname!=None : 
        print("figname: ",figname)
        plt.savefig(figname)
    plt.show()
