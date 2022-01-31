import numpy as np
from scipy import optimize
from scipy import integrate
import matplotlib.pyplot as plt


def marshall1_sub_method(tstar,k1,k2,N=10_000,p=5,correct=True) :
    l1 = 1+1/k2
    l2 = 1+1/k1
    
    output = {}
    output['tstar'] = tstar
    output['N']=N
    output['breakpoint'] = 0
    output['l1'] = l1
    output['l2'] = l2
    output['k1'] = k1
    output['k2'] = k2
    
    delta1_ = np.zeros(N+2)
    delta2_ = np.zeros(N+2)

    a_ = np.zeros(p+1)
    b_ = np.zeros(p+1)
    
    delta1_[N+1] = 1/tstar
    delta2_[N+1] = 1/tstar
    
    for j in np.arange(N+1,0,-1) : #j=N+1,...,1
        tj = tstar*j/(N+1)
                    
        #values at tj, equation (17)
        a_[0] = delta1_[j] 
        b_[0] = delta2_[j]
        
        ## updating the Taylors approximations astar_,bstar_ equations (19, 20)
        for l in range(p) :
            sum1 = np.sum([i*b_[l+1-i]*( a_[i-1]+tj*a_[i]) for i in range(1,l+1)])
            sum2 = np.sum([i*a_[l+1-i]*( b_[i-1]+tj*b_[i]) for i in range(1,l+1)])
    
            a_[l+1] = 1/((l+1)*(b_[0]-1)*tj) * ( (1/k1-(l+1)*(b_[0]-1))*a_[l]-sum1 )
            b_[l+1] = 1/((l+1)*(a_[0]-1)*tj) * ( (1/k2-(l+1)*(a_[0]-1))*b_[l]-sum2 )
        
        tjm1 = tstar*(j-1)/(N+1)
            
        delta1_[j-1] = np.polynomial.polynomial.polyval(tjm1-tj,a_) #updating at tj-1
        delta2_[j-1] = np.polynomial.polynomial.polyval(tjm1-tj,b_)
        
        if(output['breakpoint'] == 0 and ((delta1_[j-1]-l1)**2+(delta2_[j-1]-l2)**2 > ((delta1_[j]-l1)**2+(delta2_[j]-l2)**2))) :
            output['breakpoint'] = j
            if(correct) :
                break
    
    output['eps_star'] = np.sqrt(((delta1_[output['breakpoint']]-l1)**2+(delta2_[output['breakpoint']]-l2)**2))
    #output['precision'] = np.min(((delta1_-l1)**2+(delta2_-l2)**2))
    if(correct) :
        ind = output['breakpoint']
        delta1_[:ind+1] = np.linspace(l1,delta1_[ind],ind+1)
        delta2_[:ind+1] = np.linspace(l2,delta2_[ind],ind+1)
        
        #patch_ = 1/(np.linspace(0,out['tstar'],out['N']+2)[1:])
        #delta1_[1:] = np.where(delta1_[1:] > patch_, patch_,delta1_[1:])
        #delta2_[1:] = np.where(delta2_[1:] > patch_, patch_,delta2_[1:])
        
    output['delta1_']=delta1_
    output['delta2_']=delta2_
    output['values1_']=np.linspace(0,tstar,N+2)*delta1_
    output['values2_']=np.linspace(0,tstar,N+2)*delta2_
    return output

def marshall1_close_form_boundary(k1,k2,N=10_000,p=5,correct=True) :
    tstar = k1/(k1+1)
    
    if(k1!=k2) :
        tstar = 1-np.power(1+k1,k2/(k1-k2))/np.power(1+k2,k1/(k1-k2))*np.power((k2*(1+k1))/(k1*(1+k2)),k1*k2/(k1-k2))
        
    return marshall1_sub_method(tstar,k1,k2,N=N,p=p,correct=correct)

def marshall1_iterative_find_boundary_OPTIMIZE(k1,k2,method='bounded',maxiter=500,N=10_000,p=5,show_msg=False,correct=True) :
    l1 = 1+1/k2
    l2 = 1+1/k1
    
    lb = min(1/l1,1/l2)
    ub = max(1/l1,1/l2)
    
    precision_function = lambda tstar:marshall1_sub_method(tstar,k1,k2,N,p,correct=True)['eps_star']
    
    opt = optimize.minimize_scalar(precision_function,method=method,bracket=(lb,ub),bounds=(lb,ub),options={'maxiter':maxiter})
    if(show_msg) :
        print(opt)
    return marshall1_sub_method(opt.x,k1,k2,N=N,p=p,correct=correct)

def marshall1_iterative_find_boundary_CUSTOM(k1,k2,eps=10**-5,nb_eval_max=500,N=10_000,p=5,show_msg=False) :
    l1 = 1+1/k2
    l2 = 1+1/k1
    
    a = min(1/l1,1/l2)
    b = max(1/l1,1/l2)
    
    tau = (np.sqrt(5)-1)/2
    
    x_1 = a + (1-tau)*(b-a)
    f_1 = marshall1_sub_method(x_1,k1,k2,N,p,correct=True)
    
    x_2 = a + tau*(b-a)
    f_2 = marshall1_sub_method(x_2,k1,k2,N,p,correct=True)
    
    i=2
    while(f_1['eps_star']>eps and f_2['eps_star']>eps and i < nb_eval_max) :
        if(f_1['eps_star'] > f_2['eps_star']) :
            a = x_1
            x_1 = x_2
            f_1 = f_2
            x_2 = a + tau*(b-a)
            f_2 = marshall1_sub_method(x_2,k1,k2,N,p,correct=True)
        else :
            b = x_2
            x_2 = x_1
            f_2 = f_1
            x_1 = a + (1-tau)*(b-a)
            f_1 = marshall1_sub_method(x_1,k1,k2,N,p,correct=True)
        i+=1
    
    if(show_msg) :
        print("Nb of eval :",i)
        print("eps_star :",min(f_1['eps_star'],f_2['eps_star']))
        if(i==nb_eval_max) :
            print("Maximum number of evaluations reached")
            
    if(f_1['eps_star']<f_2['eps_star']) :
        return f_1
    else :
        return f_2


def marshall2_sub_method(tstar,k1,k2,N=10_000,p=5,correct=True) :
    delta1_ = np.zeros(N+2)
    delta2_ = np.zeros(N+2)
    
    l1 = 1+1/k2
    l2 = 1+1/(k1+k2-1)
    
    output = {}
    output['tstar'] = tstar
    output['N']=N
    output['breakpoint'] = 0
    output['l1'] = l1
    output['l2'] = l2
    output['k1'] = k1
    output['k2'] = k2
    
    a_ = np.zeros(p+1)
    b_ = np.zeros(p+1)
    c_ = np.zeros(p+1)
    alpha_ = np.zeros(p+1)
    beta_ = np.zeros(p+1)
    gamma_ = np.zeros(p+1)
    
    theta = (k2-1)/k1
    
    delta1_[N+1] = 1/tstar
    delta2_[N+1] = 1/tstar
    
    for j in np.arange(N+1,0,-1) : #j=N+1,...,1
        tj = tstar*j/(N+1)
                    
        #values at tj, equations (17 (modified, see 26), 36, 37)
        b_[0] = delta2_[j]
        
        #c_[0] = (delta1_[j]**(k1/(k1+k2-1)))*(delta2_[j]**((k2-1)/(k1+k2-1)))
        #a_[0] = c_[0]*(c_[0]/b_[0])**theta
        
        a_[0] = delta1_[j]
        c_[0] = np.power(a_[0],k1/(k1+k2-1))*np.power(b_[0],(k2-1)/(k1+k2-1))
        
        alpha_[0] = a_[0]*c_[0]
        beta_[0] = b_[0]*c_[0]
        gamma_[0] = a_[0]*b_[0]
        
        ## updating the Taylors approximations equations (16, 33, 34, 35)
        for l in range(p) :
            sum_b = np.sum([i*a_[l+1-i]*( b_[i-1]+tj*b_[i]) for i in range(1,l+1)])    
            b_[l+1] = -b_[l]/tj + 1/((l+1)*(a_[0]-1)*tj) * ( (1/k2)*b_[l]-sum_b )
        
            sum_c = np.sum([i*b_[l+1-i]*( c_[i-1]+tj*c_[i]) for i in range(1,l+1)])
            c_[l+1] = -c_[l]/tj + 1/((l+1)*(b_[0]-1)*tj) * ( (1/(k1+k2-1))*c_[l]-sum_c )
            
            A = np.sum([i*c_[i]*gamma_[l+1-i]-a_[i]*beta_[l+1-i] for i in range(1,l+1)]) 
            B = theta*np.sum([i*c_[i]*gamma_[l+1-i]-b_[i]*alpha_[l+1-i] for i in range(1,l+1)])
            a_[l+1] = c_[l+1]*alpha_[0]/beta_[0]+theta*(c_[l+1]*alpha_[0]/beta_[0]-b_[l+1]*alpha_[0]/beta_[0])+1/((l+1)*beta_[0]) * (A+B)
            
            alpha_[l+1] = np.sum([a_[i]*c_[l+1-i] for i in range(0,l+2)]) 
            beta_[l+1] = np.sum([b_[i]*c_[l+1-i] for i in range(0,l+2)]) 
            gamma_[l+1] = np.sum([a_[i]*b_[l+1-i] for i in range(0,l+2)]) 
        
        tjm1 = tstar*(j-1)/(N+1)
        
        delta1_[j-1] = np.polynomial.polynomial.polyval(tjm1-tj,a_) #updating at tj-1
        delta2_[j-1] = np.polynomial.polynomial.polyval(tjm1-tj,b_)
        
        if(output['breakpoint'] == 0 and ((delta1_[j-1]-l1)**2+(delta2_[j-1]-l2)**2 > ((delta1_[j]-l1)**2+(delta2_[j]-l2)**2))) :
            output['breakpoint'] = j
            if(correct) :
                break
    
    output['eps_star'] = np.sqrt(((delta1_[output['breakpoint']]-l1)**2+(delta2_[output['breakpoint']]-l2)**2))
    #output['precision'] = np.min(((delta1_-l1)**2+(delta2_-l2)**2))
    if(correct) :
        ind = output['breakpoint']
        delta1_[:ind+1] = np.linspace(l1,delta1_[ind],ind+1)
        delta2_[:ind+1] = np.linspace(l2,delta2_[ind],ind+1)
        
        #patch_ = 1/(np.linspace(0,out['tstar'],out['N']+2)[1:])
        #delta1_[1:] = np.where(delta1_[1:] > patch_, patch_,delta1_[1:])
        #delta2_[1:] = np.where(delta2_[1:] > patch_, patch_,delta2_[1:])
        
    output['delta1_']=delta1_
    output['delta2_']=delta2_
    output['values1_']=np.linspace(0,tstar,N+2)*delta1_
    output['values2_']=np.linspace(0,tstar,N+2)*delta2_
    return output
    
def marshall2_iterative_find_boundary_OPTIMIZE(k1,k2,method="bounded",maxiter=500,N=10_000,p=5,show_msg=False,correct=True) :
    l1 = 1+1/k2
    l2 = 1+1/(k1+k2-1)
    
    lb = 1/l1
    ub = 1/l2
    
    precision_function = lambda tstar:marshall2_sub_method(tstar,k1,k2,N=N,p=p,correct=True)['eps_star']
    
    opt = optimize.minimize_scalar(precision_function,method=method,bracket=(lb,ub),bounds=(lb,ub),options={'maxiter':maxiter})
    if(show_msg) :
        print(opt)
    return marshall2_sub_method(opt.x,k1,k2,N=N,p=p,correct=correct)
    
def marshall2_iterative_find_boundary_CUSTOM(k1,k2,eps=10**-5,nb_eval_max=500,N=10_000,p=5,show_msg=False) :
    l1 = 1+1/k2
    l2 = 1+1/(k1+k2-1)
    
    a = 1/l1
    b = 1/l2
    
    tau = (np.sqrt(5)-1)/2
    
    x_1 = a + (1-tau)*(b-a)
    f_1 = marshall2_sub_method(x_1,k1,k2,N,p,correct=True)
    
    x_2 = a + tau*(b-a)
    f_2 = marshall2_sub_method(x_2,k1,k2,N,p,correct=True)
    
    i=2
    while(f_1['eps_star']>eps and f_2['eps_star']>eps and i < nb_eval_max) :
        if(f_1['eps_star'] > f_2['eps_star']) :
            a = x_1
            x_1 = x_2
            f_1 = f_2
            x_2 = a + tau*(b-a)
            f_2 = marshall2_sub_method(x_2,k1,k2,N,p,correct=True)
        else :
            b = x_2
            x_2 = x_1
            f_2 = f_1
            x_1 = a + (1-tau)*(b-a)
            f_1 = marshall2_sub_method(x_1,k1,k2,N,p,correct=True)
        i+=1
    
    if(show_msg) :
        print("Nb of eval :",i)
        print("eps_star :",min(f_1['eps_star'],f_2['eps_star']))
        if(i==nb_eval_max) :
            print("Maximum number of evaluations reached")
            
    if(f_1['eps_star']<f_2['eps_star']) :
        return f_1
    else :
        return f_2


def bid1_f(v,method_out) :
    return method_out['tstar']*np.searchsorted(method_out['values1_'],v)/(method_out['N']+1)

def bid2_f(v,method_out) :
    return method_out['tstar']*np.searchsorted(method_out['values2_'],v)/(method_out['N']+1)

def display(out) :
    plt.figure(figsize=(15,4.5))
    plt.suptitle(r"Coalition $k_1={}$ vs Coalition $k_2={}$".format(out['k1'],out['k2']))
    v_ = np.linspace(0,1,1001)
        
    plt.subplot(131)
    
    plt.semilogy(np.linspace(0,out['tstar'],out['N']+2),out['delta1_'],'b',label=r'$\delta_1$')
    plt.semilogy(np.linspace(0,out['tstar'],out['N']+2),out['delta2_'],'r',label=r'$\delta_2$')
    plt.axhline(out['l1'],color='b',linestyle='--')
    plt.axhline(1/out['tstar'],color='gray',linestyle='--')
    plt.axhline(out['l2'],color='r',linestyle='--')
    plt.ylim((min(out['l2'],out['l1'])-0.1,max(out['l2'],out['l1'])+0.1))
    
    plt.axvline(out['tstar']*out['breakpoint']/(out['N']+1),color='k',linewidth=2)
    
    plt.title(r"Approximated functions : $\delta_i(b) = \lambda_i(b)/b$")
    plt.legend()
    
    plt.subplot(132)
    b_ = np.linspace(0,out['tstar'],out['N']+2)
    plt.plot(b_,out['values1_'],'b',label=r'$\lambda_1$')
    plt.plot(b_,out['values2_'],'r',label=r'$\lambda_2$')
    plt.plot(b_,out['l1']*b_,'b--')
    plt.plot(b_,out['l2']*b_,'r--')
    
    plt.title(r"Value functions $\lambda_i$")
    plt.legend()
    plt.xlim((0,out['tstar']))
    plt.grid(True, which='both')
    
    plt.subplot(133)
    plt.plot(v_,[bid1_f(v,out) for v in v_],'b',label=r"$\phi_1$")
    plt.plot(v_,[bid2_f(v,out) for v in v_],'r',label=r"$\phi_2$")
    plt.plot(v_,1/out['l1']*v_,'b--')
    plt.plot(v_,1/out['l2']*v_,'r--')
    plt.axhline(out['tstar'],color='gray',linestyle='--')
    
    plt.title(r"Bid functions $\phi_i$")
    plt.legend()
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.grid(True, which='both')
    
    plt.show()

def BR_f_OPTIMIZE(v,rv) :
    """
    Using optimize.minimize, it returns the best response function against the NON-NEGATIVE random variable rv :
    v \mapsto argmax_b (v-b)G(b)
    where G is the cdf of rv
    """
    minus_utility_v = lambda b : -(v-b)*rv.cdf(b)
    opt = optimize.minimize_scalar(minus_utility_v,method="bounded",bracket=(0,1),bounds=(0,1))
    return opt.x

#%%

def value_f(b,i,shoot_output) :
    return np.interp(b,np.linspace(0,shoot_output['tstar'],shoot_output['N']+2),shoot_output['values_'][i])

def bid_f(v,i,shoot_output) :
    return np.interp(v,shoot_output['values_'][i],np.linspace(0,shoot_output['tstar'],shoot_output['N']+2))

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

def scenario_mGU(alpha_,eps=10**-4,nb_eval_max=100,N=10_000,show_msg=False,display_graphs=False,correct=True) :
    
    m = len(alpha_)
    k_ = [np.sum(alpha_[i])  for i in range(m)]
    
    out = iterate_mGU(k_,eps=eps,nb_eval_max=nb_eval_max,N=N,show_msg=show_msg,display_graphs=display_graphs,correct=correct)
    
    b_ = np.linspace(0,out['tstar'],N+2)
    
    def integrand_coal(i,v) :
        res = v-bid_f(v,i,out)
        phi_i_v = bid_f(v,i,out)
        for j in range(m) :
            if(j!=i) :
                res *= value_f(phi_i_v,j,out)**k_[j]
        res *= k_[i]*v**(k_[i]-1)
        return res
    
    res = {'coal_':[] , 'coal__':[]}
    
    for i in range(m) :
        integrand = lambda v : integrand_coal(i,v)
        res['coal_'].append( integrate.quad(integrand, 0, 1, limit=10_000, epsabs=10**-15,epsrel=10**-15)[0] )
        res['coal__'].append( np.array(alpha_[i])/k_[i]*res['coal_'][i] )
    
    return res
    

def utility_0_f(v,n,k,o,strength_in,strength_out) :
    return (v-bid_f(v,0,o)) * v**(strength_in*(k-1)) * value_f(bid_f(v,0,o),1,o)**(strength_out*(n-k))

def utility_1_f(v,n,k,o,strength_in,strength_out) :
    return (v-bid_f(v,1,o)) * value_f(bid_f(v,1,o),0,o)**(strength_in*k) * v**(strength_out*(n-k-1))

#def utility_c_f(v,o) :
#    return (v-bid_f(v,0,o)) * value_f(bid_f(v,0,o),1,o)**(n-o['k_'][0]/strength)

def proba_0_f(v,n,k,o,strength_in,strength_out) :
    return v**(strength_in*(k-1)) * value_f(bid_f(v,0,o),1,o)**(strength_out*(n-k))

def proba_1_f(v,n,k,o,strength_in,strength_out) :
    return value_f(bid_f(v,1,o),0,o)**(strength_in*k) * v**(strength_out*(n-k-1))


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