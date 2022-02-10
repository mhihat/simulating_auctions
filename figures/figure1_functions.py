import matplotlib.pyplot as plt
import numpy as np

def value_f(b,i,shoot_output) :
    return np.interp(b,np.linspace(0,shoot_output['tstar'],shoot_output['N']+2),shoot_output['values_'][i])

def bid_f(v,i,shoot_output) :
    return np.interp(v,shoot_output['values_'][i],np.linspace(0,shoot_output['tstar'],shoot_output['N']+2))

def shoot_mGU(tstar,k_,N=10_000,correct=True) : 
    k_ = np.array(k_)
    m = len(k_)
    
    if(np.any(k_==0)) :
        print("Given coalition sizes",k_)
        print("\terror every coalition size should be != 0")
        return

    delta_ = np.zeros((m,N+2))
    
    l_ = 1+1/(np.sum(k_)-k_)
    
    output = {}
    output['m'] = m
    output['tstar'] = tstar
    output['N'] = N
    output['breakpoint'] = 0
    output['l_'] = l_
    output['k_'] = k_
    
    A_inv = 1/(m-1)*((np.ones((m,m))-np.eye(m))-(m-2)*np.eye(m))
    
    eps_f = lambda i : np.linalg.norm(delta_[:,i]-l_,2)
    
    delta_[:,N+1] = 1/tstar
    
    for j in np.arange(N+1,0,-1) : #j=N+1,...,1
        y = 1/(delta_[:,j]-1) - (np.sum(k_)-k_)
        x = A_inv@y
        
        delta_[:,j-1] = delta_[:,j]*(1-x/(j*k_))
        
        if(output['breakpoint'] == 0 and eps_f(j-1) > eps_f(j) ) :
            output['breakpoint'] = j
            if(correct) :
                break

    output['eps_star'] = eps_f(output['breakpoint'])
    if(correct) :
        ind = output['breakpoint']
        for i in range(m) :
            delta_[i,:ind+1] = np.linspace(l_[i],delta_[i,ind],ind+1)
        
    output['delta_'] = delta_
    output['values_'] = np.zeros_like(delta_)
    output['values_'] = np.linspace(0,tstar,N+2)*delta_
        
    return output

def iterate_mGU(k_,eps=10**-4,nb_eval_max=100,N=10_000,show_msg=False,display_graphs=False,correct=True) :
    if(np.any(k_==0)) :
        print("Every coalition size should be != 0")
        return
    
    k_ = np.array(k_)
    m = len(k_)
    
    l_ = 1+1/(np.sum(k_)-k_)
    
    a = 1/np.max(l_)
    b = 1/np.min(l_)
                 
    tau = (np.sqrt(5)-1)/2
    
    x_1 = a + (1-tau)*(b-a)
    f_1 = shoot_mGU(x_1,k_,N=N,correct=correct)
    
    x_2 = a + tau*(b-a)
    f_2 = shoot_mGU(x_2,k_,N=N,correct=correct)
    
    i=2
    while(f_1['eps_star']>eps and f_2['eps_star']>eps and i < nb_eval_max) :
        if(f_1['eps_star'] > f_2['eps_star']) :
            a = x_1
            x_1 = x_2
            f_1 = f_2
            x_2 = a + tau*(b-a)
            f_2 = shoot_mGU(x_2,k_,N=N,correct=correct)
        else :
            b = x_2
            x_2 = x_1
            f_2 = f_1
            x_1 = a + (1-tau)*(b-a)
            f_1 = shoot_mGU(x_1,k_,N=N,correct=correct)
        i+=1
    
    if(show_msg) :
        print("Nb of eval :",i)
        print("eps_star :",min(f_1['eps_star'],f_2['eps_star']))
        if(f_1['eps_star']<f_2['eps_star']) :
            print("tstar :",f_1['tstar'])
        else :
            print("tstar :",f_2['tstar'])
        if(i==nb_eval_max) :
            print("Maximum number of evaluations reached")
            
            
    if(f_1['eps_star']<f_2['eps_star']) :
        if(display_graphs) :
            display_mGU(f_1)
        return f_1
    else :
        if(display_graphs) :
            display_mGU(f_2)
        return f_2

def display_mGU(out) :
    plt.figure(figsize=(15,4.5))
    plt.suptitle(r"Coalitions {}".format(out['k_']))
    
    plt.subplot(131)
    v_ = np.linspace(0,1,1001)
    
    for i in range(out['m']) :
        plt.plot(np.linspace(0,out['tstar'],out['N']+2),out['delta_'][i],'b',label=r'$\delta_{}$'.format(i))
        plt.axhline(out['l_'][i],color='b',linestyle='--')
        
    plt.axhline(1/out['tstar'],color='gray',linestyle='--')
    plt.axvline(out['tstar']*out['breakpoint']/(out['N']+1),color='k',linewidth=2)
    #plt.ylim((min(out['l2_star'],out['l1_star'])-0.1,max(out['l2_star'],out['l1_star'])+0.1))
    plt.title(r"Approximated functions : $\delta_i(t) = \lambda_i(t)/t$")
    plt.legend()
    
    
    plt.subplot(132)
    b_ = np.linspace(0,out['tstar'],out['N']+2)
    
    for i in range(out['m']) :
        plt.plot(b_,out['values_'][i],'b',label=r'$\lambda_{}$'.format(i))
        plt.plot(b_,out['l_'][i]*b_,'b--')
    
    plt.axvline(out['tstar']*out['breakpoint']/(out['N']+1),color='k',linewidth=2)
    plt.title(r"Value functions $\lambda_i$")
    plt.legend()
    plt.xlim((0,out['tstar']))
    plt.grid(True, which='both')
    
    
    plt.subplot(133)    
    for i in range(out['m']) :
        plt.plot(v_,[bid_f(v,i,out) for v in v_],'b',label=r"$\phi_{}$".format(i))
        plt.plot(v_,1/out['l_'][i]*v_,'b--')
        
    plt.axhline(out['tstar'],color='gray',linestyle='--')
    plt.axhline(out['tstar']*out['breakpoint']/(out['N']+1),color='k',linewidth=2)
    plt.title(r"Bid functions $\phi_i$")
    plt.legend()
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.grid(True, which='both')
    
    plt.show()


def utility_0_f(v,n,k,o,strength_in,strength_out) :
    return (v-bid_f(v,0,o)) * v**(strength_in*(k-1)) * value_f(bid_f(v,0,o),1,o)**(strength_out*(n-k))

def utility_1_f(v,n,k,o,strength_in,strength_out) :
    return (v-bid_f(v,1,o)) * value_f(bid_f(v,1,o),0,o)**(strength_in*k) * v**(strength_out*(n-k-1))


def display_bid_functions(id,n,k,strength,strength_in,strength_out, out_):
    width=3.5
    ratio=0.9

    # plt.rcParams.update({
    # #"text.usetex": True,
    # #"font.family": "serif",
    # #"font.serif": ["Times New Roman"],
    # "font.size" : 18
    # })

    v_ = np.linspace(0,1,100)

    plt.figure(figsize=(width,width*ratio))
    plt.plot(v_,[bid_f(v,0,out_) for v in v_],'g--',label=r"$\beta^*$")
    plt.plot(v_,[bid_f(v,1,out_) for v in v_],'r-.',label=r"$\beta_*$")
    plt.plot(v_,v_*1/(1+1/(strength*(n-1))),'k-',linewidth=2,label=r"$\beta_i^{[1]}$")
    plt.title(r"Bid functions ($\alpha$={}, $n$={}, $k$={})".format(strength,n,k))
    plt.legend()
    plt.xlabel(r"value $v$")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("images/bid_graph_{}.png".format(id), dpi=300)
    plt.show()


    plt.figure(figsize=(width,width*ratio))
    plt.plot(v_,[utility_0_f(v,n,k,out_,strength_in, strength_out) for v in v_],'g--',label=r"$U_i^{[k]}$")
    plt.plot(v_,[utility_1_f(v,n,k,out_,strength_in, strength_out) for v in v_],'r-.',label=r"$U_j^{[k]}$")
    plt.plot(v_,v_**(strength*(n-1)+1)/(strength*(n-1)+1),'k-',linewidth=2,label=r"$U_j^{[1]}$")
    plt.title(r"Utility functions ($\alpha$={}, $n$={}, $k$={})".format(strength,n,k))
    plt.legend()
    plt.xlabel(r"value $v$")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("images/utility_graph_{}.png".format(id), dpi=300)
    plt.show()
