import numpy as np
import numpy.random as rand
import scipy.stats as st

def bs_formula(rands,s0,r,sigma,T):
    prices = np.zeros(len(rands))
    st = s0
    for i in range(0,len(rands)):
        prices[i] = st * np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*rands[i])
        st = prices[i]
    return prices

def arith_payoff(prices,k):
    if np.mean(prices)>k:
        return np.mean(prices)-k
    else:
        return 0.0

def geo_payoff(prices,k):
    temp = st.mstats.gmean(prices)
    if temp>k:
        return temp-k
    else:
        return 0.0

def d_plus(s0,r,sigma,T,K):
    return (np.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))

def geo_pricebybsformula(s0,r,sigma,T,K,m):
    T_bar = 0.5*(1+m)*T/m
    sigma_bar = np.sqrt(sigma**2*T/(m**2*T_bar)*(m+1)*(2*m+1)/6.0)
    delta = 0.5*(sigma**2-sigma_bar**2)
    d1 = d_plus(s0,r-delta,sigma_bar,T_bar,K)
    d2 = d1 - sigma_bar*np.sqrt(T_bar)

    return np.exp(-r*T+(r-delta)*T_bar)*s0*st.norm.cdf(d1)-np.exp(-r*T)*K*st.norm.cdf(d2)

if __name__ ==  "__main__":

# initialization
    T = 1.0
    r = 0.01   
    sigma = 0.2
    K = 100.0
    S0 = 110.0
    m = 12
    # sample size
    n = 1000
    rand.seed(11200)
    rands = rand.normal(size =(m,n))
    # generate prices
    prices = np.apply_along_axis(bs_formula,0,rands,S0,r,sigma,T/m,)

    # pricing of arithmOption
    sample_arith = np.apply_along_axis(arith_payoff,0,prices,K)
    sample_geo = np.apply_along_axis(geo_payoff,0,prices,K)

    print ('======monte_carlo=========')
    H = np.mean(sample_arith)*np.exp(-r*T)
    error = np.std(sample_arith*np.exp(-r*T))/np.sqrt(n)
    print ('price = ',H,'\n','error = ',error,'\n')
    # variance reduction
    b_star = np.cov(sample_arith, sample_geo)[0,1]/np.var(sample_geo)
    H_geo = np.mean(sample_geo)*np.exp(-r*T)
    actual_geo = geo_pricebybsformula(S0,r,sigma,T,K,m)*np.exp(-r*T)

    print ('======monte_carlo with variance reduction=========')
    H_VarRed = H - b_star*(H_geo - actual_geo)
    rho = np.corrcoef(sample_arith, sample_geo)[0,1]
    error_VarRed = error*np.sqrt((1-rho**2))
    print('price = ', H_VarRed,'\n', 'b_star = ',b_star,'\n', 'rho = ',rho,'\n','error = ', error_VarRed, '\n')