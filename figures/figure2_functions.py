import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def bid1_f(v,method_out) :
    return method_out['tstar']*np.searchsorted(method_out['values1_'],v)/(method_out['N']+1)

def bid2_f(v,method_out) :
    return method_out['tstar']*np.searchsorted(method_out['values2_'],v)/(method_out['N']+1)

def bid3_f(v,method_out) :
    return method_out['tstar']*np.searchsorted(method_out['values3_'],v)/(method_out['N']+1)



def array_to_f(x,x_,y_) :
    return y_[min(np.searchsorted(x_,x),len(x_)-1)]

def array_to_inverse_f(y,x_,y_) :
    return x_[min(np.searchsorted(y_,y),len(y_)-1)]

def shootGU(tstar,k1,k2,k3,N=10_000,correct=True) :
    delta1_ = np.zeros(N+2)
    delta2_ = np.zeros(N+2)
    delta3_ = np.zeros(N+2)
    
    l1 = (1+1/(k2+k3))*(k1!=0)
    l2 = (1+1/(k1+k3))*(k2!=0)
    l3 = (1+1/(k1+k2+k3-1))*(k3!=0)
    
    output = {}
    output['tstar'] = tstar
    output['N']=N
    output['breakpoint'] = 0
    output['l1'] = l1;    output['l2'] = l2;    output['l3'] = l3;
    output['k1'] = k1;    output['k2'] = k2;    output['k3'] = k3;
    
    eps_f = lambda i : np.sqrt((delta1_[i]-l1)**2+(delta2_[i]-l2)**2+(delta3_[i]-l3)**2)
    
    delta1_[N+1] = 1/tstar*(k1!=0)
    delta2_[N+1] = 1/tstar*(k2!=0)
    delta3_[N+1] = 1/tstar*(k3!=0)
    
    d1 = 0; d2 = 0; d3 = 0;
    d1p = 0; d2p = 0;  d3p = 0;
    
    for j in np.arange(N+1,0,-1) : #j=N+1,...,1
        tj = tstar*j/(N+1)
        tjm1 = tstar*(j-1)/(N+1)

        d1 = delta1_[j]
        d2 = delta2_[j]
        d3 = delta3_[j]
        
        #A = np.array([[0,k2*d3,k3*d2],
        #      [k1*d3,0,k3*d1],
        #      [k1*d2*d3,k2*d1*d3,(k3-1)*d1*d2]])
        #y = np.array([d2*d3/tj*( 1/(d1-1)-k2-k3 ),
        #              d1*d3/tj*( 1/(d2-1)-k1-k3 ),
        #              d1*d2*d3/tj*( 1/(d3-1) -k1-k2-k3+1) ])
        #x = np.linalg.solve(A,y)
        #d1p = x[0]; d2p = x[1]; d3p = x[2]

        #A = np.array([[0 , tj*(d1-1)*k2*d3 , tj*(d1-1)*k3*d2],
        #          [tj*(d2-1)*k1*d3 , 0 , tj*(d2-1)*k3*d1] ,
        #          [tj*(d3-1)*k1*d2*d3 , tj*(d3-1)*k2*d1*d3 , tj*(d3-1)*(k3-1)*d1*d2]])
        #y = np.array([d2*d3*( 1-(d1-1)*(k2+k3) ), d1*d3*( 1-(d2-1)*(k1+k3) ), d1*d2*d3*( 1-(d3-1)*(k1+k2+k3-1) )])
        #x = np.linalg.solve(A,y)
        #d1p = x[0]; d2p = x[1]; d3p = x[2]
        
        if(k1 != 0 and k2 != 0) :
            d1p = d1/(tj*k1) * ( (1-k3/(k3+1))/(d2-1) - k3/((k3+1)*(d1-1)) +k3/((k3+1)*(d3-1)) -k1 )
            d2p = d2/(tj*k2) * ( (1-k3/(k3+1))/(d1-1) - k3/((k3+1)*(d2-1)) +k3/((k3+1)*(d3-1)) -k2 )
            d3p = d3/(tj*(k3+1)) * ( 1/(d1-1) +1/(d2-1) -1/(d3-1) -k3-1 )
        if(k1 == 0 and k2 != 0) :
            d2p = d2/(tj*k2) * ( 1/(d3-1)-(k3-1)/(k3*(d2-1)) -k2 )
            d3p = d3/(tj*k3) * (1/(d2-1)-k3)
        if(k2 == 0 and k3 != 0) :
            d1p = d1/(tj*k1) * ( 1/(d3-1)-(k3-1)/(k3*(d1-1)) -k1 )
            d3p = d3/(tj*k3) * (1/(d1-1)-k3)
        
        delta1_[j-1] = d1+(tjm1-tj)*d1p
        delta2_[j-1] = d2+(tjm1-tj)*d2p
        delta3_[j-1] = d3+(tjm1-tj)*d3p
        
        if(output['breakpoint'] == 0 and eps_f(j-1) > eps_f(j) ) :
            output['breakpoint'] = j
            if(correct) :
                break

    output['eps_star'] = eps_f(output['breakpoint'])
    if(correct) :
        ind = output['breakpoint']
        delta1_[:ind+1] = np.linspace(l1,delta1_[ind],ind+1)
        delta2_[:ind+1] = np.linspace(l2,delta2_[ind],ind+1)
        delta3_[:ind+1] = np.linspace(l3,delta3_[ind],ind+1)
        
    output['delta1_'] = delta1_ ;   output['delta2_'] = delta2_ ;   output['delta3_'] = delta3_
    output['values1_']=np.linspace(0,tstar,N+2)*delta1_
    output['values2_']=np.linspace(0,tstar,N+2)*delta2_
    output['values3_']=np.linspace(0,tstar,N+2)*delta3_
    
    return output


def iterateGU(k1,k2,k3,eps=10**-4,nb_eval_max=100,N=10_000,show_msg=False,display_graphs=False,correct=True) :
    l1 = (1+1/(k2+k3))*(k1!=0)
    l2 = (1+1/(k1+k3))*(k2!=0)
    l3 = (1+1/(k1+k2+k3-1))*(k3!=0)
    
    a = 1/max(l1,l2,l3)
    b = 1/np.min(np.where(np.equal([l1,l2,l3],0),np.inf,[l1,l2,l3]))   
                 
    tau = (np.sqrt(5)-1)/2
    
    x_1 = a + (1-tau)*(b-a)
    f_1 = shootGU(x_1,k1,k2,k3,N=N,correct=correct)
    
    x_2 = a + tau*(b-a)
    f_2 = shootGU(x_2,k1,k2,k3,N=N,correct=correct)
    
    i=2
    while(f_1['eps_star']>eps and f_2['eps_star']>eps and i < nb_eval_max) :
        if(f_1['eps_star'] > f_2['eps_star']) :
            a = x_1
            x_1 = x_2
            f_1 = f_2
            x_2 = a + tau*(b-a)
            f_2 = shootGU(x_2,k1,k2,k3,N=N,correct=correct)
        else :
            b = x_2
            x_2 = x_1
            f_2 = f_1
            x_1 = a + (1-tau)*(b-a)
            f_1 = shootGU(x_1,k1,k2,k3,N=N,correct=correct)
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
            displayGU(f_1)
        return f_1
    else :
        if(display_graphs) :
            displayGU(f_2)
        return f_2

def displayGU(out) :
    plt.figure(figsize=(15,4.5))
    plt.suptitle(r"Coalition $k_1={}$ vs Coalition $k_2={}$ vs Individuals $k_3={}$".format(out['k1'],out['k2'],out['k3']))
    
    plt.subplot(131)
    v_ = np.linspace(0,1,1001)
    
    if(out['k1']!=0): 
        plt.plot(np.linspace(0,out['tstar'],out['N']+2),out['delta1_'],'b',label=r'$\delta_1$')
        plt.axhline(out['l1'],color='b',linestyle='--')
    if(out['k2']!=0): 
        plt.plot(np.linspace(0,out['tstar'],out['N']+2),out['delta2_'],'r',label=r'$\delta_2$')
        plt.axhline(out['l2'],color='r',linestyle='--')
    if(out['k3']!=0): 
        plt.plot(np.linspace(0,out['tstar'],out['N']+2),out['delta3_'],'g',label=r'$\delta_3$')
        plt.axhline(out['l3'],color='g',linestyle='--')
        
    plt.axhline(1/out['tstar'],color='gray',linestyle='--')
    plt.axvline(out['tstar']*out['breakpoint']/(out['N']+1),color='k',linewidth=2)
    #plt.ylim((min(out['l2_star'],out['l1_star'])-0.1,max(out['l2_star'],out['l1_star'])+0.1))
    plt.title(r"Approximated functions : $\delta_i(t) = \lambda_i(t)/t$")
    plt.legend()
    
    
    plt.subplot(132)
    b_ = np.linspace(0,out['tstar'],out['N']+2)
    
    if(out['k1']!=0): 
        plt.plot(b_,out['values1_'],'b',label=r'$\lambda_1$')
        plt.plot(b_,out['l1']*b_,'b--')
    if(out['k2']!=0): 
        plt.plot(b_,out['values2_'],'r',label=r'$\lambda_2$')
        plt.plot(b_,out['l2']*b_,'r--')
    if(out['k3']!=0):
        plt.plot(b_,out['values3_'],'g',label=r'$\lambda_3$')
        plt.plot(b_,out['l3']*b_,'g--')
    
    plt.axvline(out['tstar']*out['breakpoint']/(out['N']+1),color='k',linewidth=2)
    plt.title(r"Value functions $\lambda_i$")
    plt.legend()
    plt.xlim((0,out['tstar']))
    plt.grid(True, which='both')
    
    
    plt.subplot(133)
    if(out['k1']!=0): 
        plt.plot(v_,[bid1_f(v,out) for v in v_],'b',label=r"$\phi_1$")
        plt.plot(v_,1/out['l1']*v_,'b--')
    if(out['k2']!=0) :
        plt.plot(v_,[bid2_f(v,out) for v in v_],'r',label=r"$\phi_2$")
        plt.plot(v_,1/out['l2']*v_,'r--')
    if(out['k3']!=0):
        plt.plot(v_,1/out['l3']*v_,'g--')
        plt.plot(v_,[bid3_f(v,out) for v in v_],'g',label=r"$\phi_3$")
        
    plt.axhline(out['tstar'],color='gray',linestyle='--')
    plt.axhline(out['tstar']*out['breakpoint']/(out['N']+1),color='k',linewidth=2)
    plt.title(r"Bid functions $\phi_i$")
    plt.legend()
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.grid(True, which='both')
    
    plt.show()
    
def scenarioGU_1(ks,kw,alpha,eps=10**-4,nb_eval_max=100,N=10_000,show_msg=False,display_graphs=False,correct=True) :
    k1 = ks*alpha
    k2 = 0
    k3 = kw
    
    out = iterateGU(k1,k2,k3,eps=eps,nb_eval_max=nb_eval_max,N=N,show_msg=show_msg,display_graphs=display_graphs,correct=correct)
    
    b_ = np.linspace(0,out['tstar'],N+2)
    
    v1 = lambda b:array_to_f(b,b_,out['values1_'])
    #v2 = lambda b:array_to_f(b,b_,out['values2_'])
    v3 = lambda b:array_to_f(b,b_,out['values3_'])
    
    b1 = lambda v:array_to_inverse_f(v,b_,out['values1_'])
    #b2 = lambda v:array_to_inverse_f(v,b_,out['values2_'])
    b3 = lambda v:array_to_inverse_f(v,b_,out['values3_'])
    
    integrand_coal = lambda v : (v-b1(v)) * v3(b1(v))**k3 * k1*v**(k1-1)
    integrand_ext = lambda v : (v-b3(v)) * v1(b3(v))**k1 * k3*v**(k3-1)
    
    res = {}
    res['coal_sum'] = integrate.quad(integrand_coal, 0, 1, limit=2000, epsabs=10**-11,epsrel=10**-11)[0]
    res['coal_ind'] = res['coal_sum']/ks
    
    res['ext_sum'] = integrate.quad(integrand_ext, 0, 1, limit=2000, epsabs=10**-11,epsrel=10**-11)[0]
    res['ext_ind'] = res['ext_sum']/kw
    return res

def scenarioGU_2(ks,kw,alpha,eps=10**-4,nb_eval_max=100,N=10_000,show_msg=False,display_graphs=False,correct=True) :
    k1 = (ks-1)*alpha
    k2 = 1*alpha
    k3 = kw
    
    out = iterateGU( k1, k2, k3, eps=eps,nb_eval_max=nb_eval_max,N=N,show_msg=show_msg,display_graphs=display_graphs,correct=correct)
    
    b_ = np.linspace(0,out['tstar'],N+2)
        
    v1 = lambda b:array_to_f(b,b_,out['values1_'])
    v2 = lambda b:array_to_f(b,b_,out['values2_'])
    v3 = lambda b:array_to_f(b,b_,out['values3_'])
    
    b1 = lambda v:array_to_inverse_f(v,b_,out['values1_'])
    b2 = lambda v:array_to_inverse_f(v,b_,out['values2_'])
    b3 = lambda v:array_to_inverse_f(v,b_,out['values3_'])
    
    integrand_coal = lambda v : (v-b1(v)) * v2(b1(v))**k2 * v3(b1(v))**k3 * k1*v**(k1-1)
    integrand_strong = lambda v : (v-b2(v)) * v1(b2(v))**k1 * v3(b2(v))**k3 *  k2*v**(k2-1)
    integrand_ext = lambda v : (v-b3(v)) * v1(b3(v))**k1 * v2(b3(v))**k2 * k3*v**(k3-1)
    
    res = {}
    res['coal_sum'] = integrate.quad(integrand_coal, 0, 1, limit=2000, epsabs=10**-11,epsrel=10**-11)[0]
    res['coal_ind'] = res['coal_sum']/(ks-1)
    
    res['strong_ind'] = integrate.quad(integrand_strong, 0, 1, limit=2000, epsabs=10**-11,epsrel=10**-11)[0]
    
    res['ext_sum'] = integrate.quad(integrand_ext, 0, 1, limit=2000, epsabs=10**-11,epsrel=10**-11)[0]
    res['ext_ind'] = res['ext_sum']/kw
    return res
