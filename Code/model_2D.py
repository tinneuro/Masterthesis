# importing python modules
import numpy as np
import matplotlib.pyplot as plt
import unidip.dip as dip
# importing own functions
import statistics as stats_UD


# setting basic parameters
# units seconds or Hz
tau = 0.01
tau_a = 1
dt = 0.001
r_max = 10
N = 600


def sigmoidal_transfer(h,g,theta):
    """Sigmoidal transfer function."""
    phi = r_max/(1 + np.exp(-g*(h-theta)))
    return phi

def Sys(X,t,params):
    """The 2D differential equations system to integrate analytically"""
    # X[0] = h, X[1] = a
    beta,w,I,g,theta = params
    F = sigmoidal_transfer(X[0],g,theta)
    h = 1/tau*(-X[0] + w*F + I - X[1])
    a = 1/tau_a*(-X[1] + beta*F)

    return np.array([h,a])


def pop_activity(r,N):
    """Computing the population activity using the Poisson distribution to create some noise"""
    pois = np.random.poisson(N*r*dt)
    A = pois/(N*dt)
    return A

def f_poisson(h,a,transfer_function,params_sigmoid,params_weights):
    """Original rhs for 2D model with Poisson noise and with noise in the adaptation as well"""
    g,theta = params_sigmoid
    beta,w,I = params_weights

    r = transfer_function(h,g,theta)
    A = pop_activity(r,N)

    f_h = 1/tau*(-h + w*A + I -a)
    f_h_det = 1/tau*(-h + w*r + I -a)
    #f_h_noise = (A-r)
    #f_h = f_h_det + w/tau*f_h_noise
    f_a = 1/tau_a*(-a + beta*A)

    return f_h,f_a,r,f_h_det#,f_h_noise

def f_poisson_same_format(h,a,transfer_function,params_sigmoid,params_weights,*args):
    """Poisson noise such that it is in the format of the other f_functions to be used in the run_sim_2D"""
    g,theta = params_sigmoid
    beta,w,I = params_weights

    sqrtdt = np.sqrt(dt)

    r = transfer_function(h,g,theta)
    A = pop_activity(r,N)

    f_h_det = 1/tau*(-h + w*r + I - a) # deterministic dynamics
    f_h_noise = sqrtdt*(A-r)# need the sqrtdt to have dt in front of the (A-r) later due to mult with sqrtdt later
    f_a = 1/tau_a*(-a + beta*r)

    return f_h_det,f_h_noise,f_a,r

def f_poisson_no_noise_adapt(h,a,transfer_function,params_sigmoid,params_weights):
    """Poisson multiplicative noise with no noise in Adaptation"""
    g,theta = params_sigmoid
    beta,w,I = params_weights

    r = transfer_function(h,g,theta)
    A = pop_activity(r,N)

    f_h_det = 1/tau*(-h + w*r + I -a)
    f_h = f_h_det + w/tau*(A-r)
    f_a = 1/tau_a*(-a + beta*r)

    return f_h,f_a,r,f_h_det

def f_mult_gauss_noise(h,a,transfer_function,params_sigmoid,params_weights,*args):
    """Gaussian multiplicative noise. This function can both be used for noise in adaptation case or no noise in adaptation case.
    Depends which run_sim_2D_grid function one uses."""
    g,theta = params_sigmoid
    beta,w,I = params_weights

    r = transfer_function(h,g,theta)
    sqrtr = np.sqrt(r)

    xi = np.random.randn() # gaussian white noise sqrtdt Euler-Maryuama in the function run_sim_2D

    f_h_det = 1/tau*(-h + w*r + I - a) # deterministic r-dynamics
    f_h_noise = sqrtr/sqrtN*xi
    f_a = 1/tau_a*(-a + beta*r)# deterministic a-dynamics

    return f_h_det,f_h_noise,f_a,r


def f_add_noise(h,a,transfer_function,params_sigmoid,params_weights,sigma):
    """Additive gaussian white noise using parameter noise strength sigma.
    The noise strength can vary over the grid using Kramers-Moyal but it can also stay constant over the grid."""
    g,theta = params_sigmoid
    beta,w,I = params_weights

    r = transfer_function(h,g,theta)

    xi = np.random.randn() # gaussian white noise sqrtdt Euler-Maryuama in the function run_sim_2D

    f_h_det = 1/tau*(-h + w*r + I - a) # deterministic dynamics
    f_h_noise = sigma*xi
    f_a = 1/tau_a*(-a + beta*r)

    return f_h_det,f_h_noise,f_a,r

def f_add_noise_w_beta_inside_sigma(h,a,transfer_function,params_sigmoid,params_weights,sigma_w):
    """Additive gaussian white noise using parameter noise strength sigma estimated via Kramer-Moyal
    We here have a constant noise strength over the course of the total simulation time."""
    g,theta = params_sigmoid
    beta,w,I = params_weights

    r = transfer_function(h,g,theta)

    xi = np.random.randn() # gaussian white noise sqrtdt Euler-Maryuama in the function run_sim_2D

    f_h_det = 1/tau*(-h + w*r + I - a) # deterministic dynamics
    f_h_noise_w = sigma_w*xi
    f_a = 1/tau_a*(-a + beta*r)

    return f_h_det,f_h_noise_w,f_a,r

def run_sim_2D_poisson(stop_time,transfer_function,f_rhs,params_sigmoid,params_weights):
    """Different simulation function for the cases where we have poisson noise.
    There I do not split into the noise part and the deterministic part.
    Uses f_poisson"""
    # Initialisation
    # time grid
    sim_time = np.arange(0,stop_time,dt)
    sim_length = len(sim_time)
    beta,w,I = params_weights

    # storage of he,a and the rate
    h = np.zeros((sim_length))
    a = np.zeros((sim_length))
    rate = np.zeros((sim_length))

    kramer_sig = np.zeros((sim_length-1))
    for i in range(sim_length-1):
    # Run the simulation using Eulers method
        # computation of rhs
        f_h,f_a,r,f_h_det= f_rhs(h[i],a[i],transfer_function,params_sigmoid,params_weights)
        # Perform Euler step
        h[i+1] = h[i] + dt*f_h
        a[i+1] = a[i] + dt*f_a

        rate[i+1] = r
        kramer_sig[i] = tau/w*(h[i+1]-h[i]-dt*f_h_det)


    return sim_time,h,a,rate,kramer_sig



def run_sim_2D(stop_time,transfer_function,f_rhs,params_sigmoid,params_weights,*args):
    """Run simulation with noise in excitation only.
    Perform the Euler step calling f_rhs for the right hand side of the SDE.
    f_rhs can be poisson, gauss multiplicative or gauss additive noise.(f_poisson_same_format,f_mult_gauss_noise,f_add_gauss_noise)
    We store the increments for Kramers-Moyal coefficient in an array to be able to plot the distribution later.
    *args parameter is the sigma in the case that we want to simulate the additive noise model.
    This sigma is without the parameter w and beta, these parameters are multiplied with sigma in the equations below"""
    global sqrtN  # need to put these variables inside here to recompute if N or dt is changed in Notebook

    sqrtdt = np.sqrt(dt)
    sqrtN = np.sqrt(N)
    beta,w,I = params_weights
    # Initialisation
    # time grid
    sim_time = np.arange(0,stop_time,dt)
    sim_length = len(sim_time)

    # storage of he,a and the rate
    h = np.zeros((sim_length))
    a = np.zeros((sim_length))
    rate = np.zeros((sim_length))
    kramer_sig = np.zeros((sim_length-1)) # storing time series to be able to plot difference in distribution in Up versus Down
    #kramer_sig_check = np.zeros((sim_length-1))

    for i in range(sim_length-1):
    # Run the simulation using Eulers method
        # computation of rhs
        f_h_det,f_h_noise,f_a,r = f_rhs(h[i],a[i],transfer_function,params_sigmoid,params_weights,*args)
        # Perform Euler step
        sqrt_noise = sqrtdt*f_h_noise
        h[i+1] = h[i] + dt*f_h_det + w/tau*sqrt_noise
        a[i+1] = a[i] + dt*f_a

        rate[i+1] = r
        kramer_sig[i] = f_h_noise
        #kramer_sig_check[i] = (tau*(h[i+1]-h[i])- dt*tau*f_h_det) # checking if std of distribution gives kramers-coefficient

    #kramer_sig_check_std = 1/sqrtdt*np.std(kramer_sig_check)

    return sim_time,h,a,rate,kramer_sig

def run_sim_2D_adapt_noise(stop_time,transfer_function,f_rhs,params_sigmoid,params_weights,*args):
    """ Run simulation with noise in excitation AND adaptation.
    Perform the Euler step calling f_rhs for the right hand side of the SDE.
    f_rhs can be poisson, gauss multiplicative or gauss additive noise.(f_poisson_same_format,f_mult_gauss_noise,f_add_gauss_noise)
    We store the increments for Kramers-Moyal coefficient in an array to be able to plot the distribution later.
    *args parameter is the sigma in the case that we want to simulate the additive noise model.
    This sigma is without the parameter w and beta, these parameters are multiplied with sigma in the equations below"""
    global sqrtN  # need to put these variables inside here to recompute if N or dt is changed in Notebook

    sqrtdt = np.sqrt(dt)
    sqrtN = np.sqrt(N)
    beta,w,I = params_weights
    # Initialisation
    # time grid
    sim_time = np.arange(0,stop_time,dt)
    sim_length = len(sim_time)

    # storage of he,a and the rate
    h = np.zeros((sim_length))
    a = np.zeros((sim_length))
    rate = np.zeros((sim_length))
    kramer_sig = np.zeros((sim_length-1)) # storing time series to be able to plot difference in distribution in Up versus Down

    for i in range(sim_length-1):
    # Run the simulation using Eulers method
        # computation of rhs
        f_h_det,f_h_noise,f_a,r = f_rhs(h[i],a[i],transfer_function,params_sigmoid,params_weights,*args)
        # Perform Euler step
        sqrt_noise = sqrtdt*f_h_noise
        h[i+1] = h[i] + dt*f_h_det + w/tau*sqrt_noise
        a[i+1] = a[i] + dt*f_a + beta/tau_a*sqrt_noise

        rate[i+1] = r
        kramer_sig[i] = f_h_noise

    return sim_time,h,a,rate,kramer_sig

def run_sim_2D_grid(stop_time,transfer_function,f_rhs,params_sigmoid,params_weights,*args):
    """Same function as run_sim_2D but efficient computation of Kramer-Moyal via running average.
    Do not store each increment in the array.
    No noise in the adaptation."""
    global sqrtN  # need to put these variables inside here to recompute if N or dt is changed in Notebook

    sqrtdt = np.sqrt(dt)
    sqrtN = np.sqrt(N)
    beta,w,I = params_weights
    # Initialisation
    # time grid
    sim_time = np.arange(0,stop_time,dt)
    sim_length = len(sim_time)

    # storage of he,a and the rate
    h = np.zeros((sim_length))
    a = np.zeros((sim_length))
    rate = np.zeros((sim_length))

    # online kramer-moyal average initialisation
    mean = 0
    n = 0

    for i in range(sim_length-1):
    # Run the simulation using Eulers method
        # computation of rhs
        f_h_det,f_h_noise,f_a,r = f_rhs(h[i],a[i],transfer_function,params_sigmoid,params_weights,*args)
        # Perform Euler step
        sqrt_noise = sqrtdt*f_h_noise
        h[i+1] = h[i] + dt*f_h_det + w/tau*sqrt_noise
        a[i+1] = a[i] + dt*f_a

        rate[i+1] = r
        n += 1
        mean = mean + (f_h_noise**2 - mean)/n # computing mean

    kramer_sig = np.sqrt(mean) # standard deviation

    return sim_time,h,a,rate,kramer_sig

def run_sim_2D_grid_adapt_noise(stop_time,transfer_function,f_rhs,params_sigmoid,params_weights,*args):
    """Same function as run_sim_2D_adapt_noise but efficient computation of Kramer-Moyal via running average.
    Do not store each increment in the array.
    Noise in the excitation and adaptation."""
    global sqrtN  # need to put these variables inside here to recompute if N or dt is changed in Notebook

    sqrtdt = np.sqrt(dt)
    sqrtN = np.sqrt(N)
    beta,w,I = params_weights
    # Initialisation
    # time grid
    sim_time = np.arange(0,stop_time,dt)
    sim_length = len(sim_time)

    # storage of he,a and the rate
    h = np.zeros((sim_length))
    a = np.zeros((sim_length))
    rate = np.zeros((sim_length))

    # online kramer-moyal average initialisation
    mean = 0
    n = 0

    for i in range(sim_length-1):
    # Run the simulation using Eulers method
        # computation of rhs
        f_h_det,f_h_noise,f_a,r = f_rhs(h[i],a[i],transfer_function,params_sigmoid,params_weights,*args)
        # Perform Euler step
        sqrt_noise = sqrtdt*f_h_noise
        h[i+1] = h[i] + dt*f_h_det + w/tau*sqrt_noise
        a[i+1] = a[i] + dt*f_a + beta/tau_a*sqrt_noise

        rate[i+1] = r
        n += 1
        mean = mean + (f_h_noise**2 - mean)/n # computing mean

    kramer_sig = np.sqrt(mean) # standard deviation

    return sim_time,h,a,rate,kramer_sig

def run_sim_2D_grid_adapt_noise_smalldt(stop_time,integration_step,transfer_function,f_rhs,params_sigmoid,params_weights,*args):
    """Function to check smaller dt.
    Only storing each integration_step datapoint such that we store exactly the same amount of points as for the bigger dt.
    integration_step is an integer n telling that every n-th datapoint should be stored."""
    global sqrtN  # need to put these variables inside here to recompute if N or dt is changed in Notebook

    sqrtdt = np.sqrt(dt)
    sqrtN = np.sqrt(N)
    beta,w,I = params_weights
    # Initialisation
    # time grid
    sim_time = np.arange(0,stop_time,dt)
    sim_length = len(sim_time)

    storage_length = int(len(sim_time)/integration_step)

    # storage of he,a and the rate
    sim_time_store = np.arange(0,stop_time,integration_step*dt)
    h = np.zeros((storage_length))
    a = np.zeros((storage_length))
    f_h = np.zeros((storage_length))
    rate = np.zeros((storage_length))

    # online kramer-moyal average initialisation
    mean_check = 0 # mean to check it is really zero we can use the E(X**2) formula
    mean = 0
    n = 0

    # initialisation of first h and a
    h_prev = 0
    a_prev = 0

    for i in range(sim_length-1):
    # Run the simulation using Eulers method for the small timestep but only store the result for every integration_step
        # computation of rhs
        f_h_det,f_h_noise,f_a,r = f_rhs(h_prev,a_prev,transfer_function,params_sigmoid,params_weights,*args)
        # Perform Euler step
        sqrt_noise = sqrtdt*f_h_noise
        h_next = h_prev + dt*f_h_det + w/tau*sqrt_noise
        a_next = a_prev + dt*f_a + beta/tau_a*sqrt_noise

        h_prev = h_next
        a_prev = a_next

        if i == 0:
            f_h[0] = f_h_det

        # store the result every integration_step but not for the first
        if (i % integration_step == 0) & (i != 0):
            j = int(i/integration_step) # starts with 1 for i = integration_step
            h[j] = h_next
            a[j] = a_next
            f_h[j] = f_h_det

            rate[j] = r
            noise = (h[j] - h[j-1] - dt*integration_step*f_h[j-1])
            n += 1
            mean_check = mean_check + (noise - mean_check)/n
            mean = mean + (noise**2 - mean)/n # computing mean

    kramer_sig = tau/w*1/np.sqrt(dt*integration_step)*np.sqrt(mean-mean_check**2) # standard deviation std(X) = sqrt(Var(X)) = sqrt(E(X^2)-E(X)^2)

    return sim_time_store,h,a,rate,kramer_sig#,mean_check

def run_sim_2D_grid_only_adapt_noise(stop_time,transfer_function,f_rhs,params_sigmoid,params_weights,*args):
    """Run the 2D simulation with efficient Kramer-Moyal estimation and with mutliplicative noise ONLY in adaptation variable.
    This is just trying out things for understanding noise in adaptation."""
    global sqrtN  # need to put these variables inside here to recompute if N or dt is changed in Notebook

    sqrtdt = np.sqrt(dt)
    sqrtN = np.sqrt(N)
    beta,w,I = params_weights
    # Initialisation
    # time grid
    sim_time = np.arange(0,stop_time,dt)
    sim_length = len(sim_time)

    # storage of he,a and the rate
    h = np.zeros((sim_length))
    a = np.zeros((sim_length))
    rate = np.zeros((sim_length))

    # online kramer-moyal average initialisation
    mean = 0
    n = 0

    for i in range(sim_length-1):
    # Run the simulation using Eulers method
        # computation of rhs
        f_h_det,f_h_noise,f_a,r = f_rhs(h[i],a[i],transfer_function,params_sigmoid,params_weights,*args)
        # Perform Euler step
        sqrt_noise = sqrtdt*f_h_noise
        h[i+1] = h[i] + dt*f_h_det
        a[i+1] = a[i] + dt*f_a + beta/tau_a*sqrt_noise

        rate[i+1] = r
        n += 1
        mean = mean + (f_h_noise**2 - mean)/n # computing mean

    kramer_sig = np.sqrt(mean) # standard deviation

    return sim_time,h,a,rate,kramer_sig

def run_sim_2D_grid_noise_constant(stop_time,transfer_function,f_rhs,params_sigmoid,params_weights,sigma_w):
    """Run the 2D simulation with noise in excitation only and noise strength sigma_w already includes w.
    Input argument f_rhs is not needed, just added such that format is the same as in the other functions."""
    # previous version also had mutliplicative noise in adaptation variable."""
    global sqrtN  # need to put these variables inside here to recompute if N or dt is changed in Notebook

    sqrtdt = np.sqrt(dt)
    sqrtN = np.sqrt(N)
    beta,w,I = params_weights
    # Initialisation
    # time grid
    sim_time = np.arange(0,stop_time,dt)
    sim_length = len(sim_time)

    # storage of he,a and the rate
    h = np.zeros((sim_length))
    a = np.zeros((sim_length))
    rate = np.zeros((sim_length))

    # online kramer-moyal average initialisation
    mean = 0
    n = 0

    for i in range(sim_length-1):
    # Run the simulation using Eulers method
        # computation of rhs
        f_h_det,f_h_noise_w,f_a,r = f_add_noise_w_beta_inside_sigma(h[i],a[i],transfer_function,params_sigmoid,params_weights,sigma_w)
        # Perform Euler step
        sqrt_noise_w = sqrtdt*f_h_noise_w
        h[i+1] = h[i] + dt*f_h_det + 1/tau*sqrt_noise_w
        a[i+1] = a[i] + dt*f_a

        rate[i+1] = r
        n += 1
        mean = mean + (f_h_noise_w**2 - mean)/n # computing mean

    kramer_sig = np.sqrt(mean) # standard deviation

    return sim_time,h,a,rate,kramer_sig

def compute_mean_stderror(result_trials):
    """Computation of standard error on the mean"""
    ntrials = len(result_trials)
    mean = np.mean(result_trials)
    stderror = np.std(result_trials)/np.sqrt(ntrials)

    return mean,stderror

def run_sim_2D_trials(ntrials,sim_len,run_sim_fct,f_rhs,*sig_e):
    """Run several trials of the 2D simulation.
     Compute the mean of the individual statistics and the standard error on the mean"""

    perc_trials_up = []
    perc_trials_down = []
    mean_UP_trials = []
    mean_DOWN_trials = []
    cv_UP_trials = []
    cv_DOWN_trials = []
    kramer_sig_trials = []

    lag0_trials = []
    lag1_trials = []

    for i in range(ntrials):
        results_stats,kramer_sig = get_stats_of_model(sim_len,run_sim_fct,f_rhs,params_sigmoid,params_weights,*sig_e)
        percUP,percDOWN,mean_up,cv_up,mean_down,cv_down,skew_up,skew_down,scc_lag0,scc_lag1 = results_stats

        perc_trials_up.append(percUP)
        perc_trials_down.append(percDOWN)

        mean_UP_trials.append(mean_up)
        mean_DOWN_trials.append(mean_down)
        cv_UP_trials.append(cv_up)
        cv_DOWN_trials.append(cv_down)

        lag0_trials.append(scc_lag0)
        lag1_trials.append(scc_lag1)

        kramer_sig_trials.append(kramer_sig)

        print(f"Trial {i+1} done")

    percUP_mean,percUP_stderror = compute_mean_stderror(perc_trials_up)
    percDOWN_mean,percDOWN_stderror = compute_mean_stderror(perc_trials_down)
    meanUP_mean,meanUP_stderror = compute_mean_stderror(mean_UP_trials)
    meanDOWN_mean,meanDOWN_stderror = compute_mean_stderror(mean_DOWN_trials)
    cvUP_mean,cvUP_stderror = compute_mean_stderror(cv_UP_trials)
    cvDOWN_mean,cvDOWN_stderror = compute_mean_stderror(cv_DOWN_trials)

    scc_lag0_mean,scc_lag0_stderror = compute_mean_stderror(lag0_trials)
    scc_lag1_mean,scc_lag1_stderror = compute_mean_stderror(lag1_trials)

    kramer_sig_mean,kramer_sig_stderror = compute_mean_stderror(kramer_sig_trials)

    results = (percUP_mean,percUP_stderror,percDOWN_mean,percDOWN_stderror,meanUP_mean,meanUP_stderror,meanDOWN_mean,meanDOWN_stderror,cvUP_mean,cvUP_stderror,cvDOWN_mean,cvDOWN_stderror,scc_lag0_mean,scc_lag0_stderror,scc_lag1_mean,scc_lag1_stderror)

    return results,kramer_sig_mean,kramer_sig_stderror


def exc_nullcline_2D(h,params_sigmoid,params_weights):
    """Calculating the nullcline for the h variable"""
    g,theta = params_sigmoid
    beta,w,I = params_weights

    F = sigmoidal_transfer(h,g,theta)
    a = -h + w*F + I
    return a

def adapt_nullcline_2D(h,params_sigmoid,params_weights):
    """Calculating the nullcline for adaptation"""
    g,theta = params_sigmoid
    beta,w,I = params_weights

    F = sigmoidal_transfer(h,g,theta)
    a = beta*F
    return a

def interactive_nullclines_h_dynamics(beta = 0.4,w = 2.4,I = 0,g = 0.25,theta = 10):
    """Interactive Nullclines Plot with Sliders for the dynamics in h"""
    params = {'figure.figsize': (10,8),
      'lines.linewidth': 3,
      'legend.fontsize': 20,
     'axes.labelsize': 20,
     'axes.titlesize':20,
     'xtick.labelsize':20,
     'ytick.labelsize':20,
     'figure.constrained_layout.use': False}
    plt.rcParams.update(params)
    params_sigmoid = [g,theta]
    params_weights = [beta,w,I]
    h = np.linspace(-10,40,200)

    h_nullcline = exc_nullcline_2D(h,params_sigmoid,params_weights)
    a_nullcline = adapt_nullcline_2D(h,params_sigmoid,params_weights)

    plt.plot(h,h_nullcline,label = "r Nullcline")
    plt.plot(h,a_nullcline,label = "a Nullcline")
    plt.xlabel("Input potential in mV")
    plt.ylabel("Adaptation mV")
    plt.xlim(-10,40)
    plt.ylim(-10,10)
    plt.legend()
    plt.show()

def interactive_nullclines_r_dynamics(beta = 0.4,w = 2.4,I = 0,g = 0.25,theta = 10):
    """Interactive Nullclines Plot with Sliders for the dynamics in r"""
    params = {'figure.figsize': (10,8),
      'lines.linewidth': 3,
      'legend.fontsize': 20,
     'axes.labelsize': 20,
     'axes.titlesize':20,
     'xtick.labelsize':20,
     'ytick.labelsize':20,
     'figure.constrained_layout.use': False}
    plt.rcParams.update(params)
    params_sigmoid = [g,theta]
    params_weights = [beta,w,I]
    h = np.linspace(-10,40,200)
    F = sigmoidal_transfer(h,g,theta)

    h_nullcline = exc_nullcline_2D(h,params_sigmoid,params_weights)
    a_nullcline = adapt_nullcline_2D(h,params_sigmoid,params_weights)

    plt.plot(F,h_nullcline,label = "r Nullcline",color = "#348ABD")
    plt.plot(F,a_nullcline,label = "a Nullcline",color = "darkorange")
    plt.xlabel("Population rate Hz")
    plt.ylabel("Adaptation mV")
    plt.xlim(0,10)
    plt.ylim(-5,10)
    plt.legend()
    plt.show()


def mov_avg_sim_time_rate_cut_off(x,sim_time,window_avg,start_cut,end_cut):
    """Moving average + cutting of beginning and end
    x is firing rate, sim_time is the time, window_avg is the moving average window size,
    start_cut is how much we cut off from the start, end_cut how much we cut off from the end."""
    cut_off_start = int(start_cut/dt)
    cut_off_end = int(end_cut/dt)
    mov_avg = stats_UD.moving_average(x,window_avg)
    mov_avg_cut = mov_avg[cut_off_start:-cut_off_end]
    sim_time_cut = sim_time[cut_off_start:-cut_off_end]
    rate_cut = x[cut_off_start:-cut_off_end]

    return sim_time_cut,rate_cut,mov_avg_cut

def compute_stats_helper(sim_time,rate):
    """Compute all statistics we need from simulation time and firing rate.
    1) Computes the moving average and cut off the first second and last 500ms
    2) Find thresholds based on the moving average using stats_UD.quartic_polynomial_fit
    3) Classify into Up and Down or no Up and Down transitions present
    3) If transitions present compute the statistics of Up and Downs
    """
    # plotting
    start_time = 60
    end_time = 80

    sim_time_cut,rate_cut,mov_avg_cut = mov_avg_sim_time_rate_cut_off(rate,sim_time,100,1,0.5)# calculate mov_avg from cut off rate and sim time
    #every_n_point = int(len(mov_avg_cut)/7000) # pvalue data reduction
    #mov_avg_less_data = np.msort(mov_avg_cut)[::every_n_point]
    #_,pval,intervals = dip.diptst(mov_avg_less_data)

    # compute the thresholds from the data
    threshs,bimodal = stats_UD.quartic_polynomial_fit(mov_avg_cut,50)
    #print("pvalue",np.round(pval,3))
    #print("variance of firing rate",np.round(np.var(rate_cut),3))
    if bimodal: # bimodal distribution, if there are thresholds found and the histogram has bigger support than 5
        print("Up and Down Transitions present!")
        threshDOWN,threshUP = threshs
        # compute crossings
        crossings,start_down,UP_dur,DOWN_dur = stats_UD.crossings_durations_2thresh(sim_time_cut,mov_avg_cut,threshUP,threshDOWN)# compute crossings over the thresholds
        if UP_dur.size != 0 or DOWN_dur.size != 0:# we want to have non empty lists for Up and Down durations, can happen when thresholds are badly chosen
            percUP,percDOWN = stats_UD.ratio_simtime_up_down(UP_dur,DOWN_dur)# compute percUp percDown
            mean_up,cv_up,mean_down,cv_down,skew_up,skew_down = stats_UD.mean_cv_skew(UP_dur,DOWN_dur)# compute CV and mean UP & DOWN
            if len(UP_dur) >= 400:#check that we have enough transitions for SCC!
                if len(UP_dur) == len(DOWN_dur):# equal amount of Ups and Downs
                    print("Len UP/DOWN dur equal len",len(UP_dur),len(DOWN_dur))
                    serial_corr = stats_UD.compute_scc(0,1,UP_dur,DOWN_dur) # computing SCC for lag 0 and 1
                    scc_lag0,scc_lag1 = serial_corr # lag 0 D->U, lag1 U->D

                else:# unequal lengths of Ups and Downs
                    print("Len UP/DOWN dur no equal len",len(UP_dur),len(DOWN_dur))
                    serial_corr = stats_UD.compute_scc(0,1,UP_dur,DOWN_dur[:-1])
                    scc_lag0,scc_lag1 = serial_corr

                results = (percUP,percDOWN,mean_up,cv_up,mean_down,cv_down,skew_up,skew_down,scc_lag0,scc_lag1)
            else:# to few transitions to properly calculate SCC!
                results = (percUP,percDOWN,mean_up,cv_up,mean_down,cv_down,skew_up,skew_down,np.nan,np.nan)

        else:
            print("No Up and Down transitions: No Crossings!")
            percUP,percDOWN,mean_up,cv_up,mean_down,cv_down,skew_up,skew_down,scc_lag0,scc_lag1 = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
            results = (percUP,percDOWN,mean_up,cv_up,mean_down,cv_down,skew_up,skew_down,scc_lag0,scc_lag1)

        return results
    else:
        print("No Up and Down transitions!")
        #plt.plot(sim_time_cut[int(start_time/dt):int(end_time/dt)],rate_cut[int(start_time/dt):int(end_time/dt)],label = "firing rate",linewidth = 2)
        #plt.plot(sim_time_cut[int(start_time/dt):int(end_time/dt)],mov_avg_cut[int(start_time/dt):int(end_time/dt)],label = "moving average",linewidth = 2)
        #plt.legend()
        #plt.show()
        percUP,percDOWN,mean_up,cv_up,mean_down,cv_down,skew_up,skew_down,scc_lag0,scc_lag1 = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        results = (percUP,percDOWN,mean_up,cv_up,mean_down,cv_down,skew_up,skew_down,scc_lag0,scc_lag1)
        return results

def get_stats_of_model(sim_len,f_sim,f_rhs,params_sigmoid,params_weights,*sig_e):
    """Compute for one parametersetting of params_weights the statistics of Up and Down duration for one trial of sim_len.
    This function runs the simulation and then calls compute_stats_helper to compute the statistics.
    Depending on the choice of f_sim there can be noise in the adaptation or not.(f_sim e.g.: run_sim_2D_grid or run_sim_2D_grid_noise_constant or run_sim_2D)
    Depending on f_rhs the multiplicative or additive noise model can be used. (f_rhs e.g.: f_add_noise or f_mult_gauss_noise)"""

    sim_time,h,a,rate,kramer_sig = f_sim(sim_len,sigmoidal_transfer,f_rhs,params_sigmoid,params_weights,*sig_e)
    results = compute_stats_helper(sim_time,rate)

    return results,kramer_sig

def get_stats_of_model_smalldt(sim_len,f_rhs,params_sigmoid,params_weights,int_step,*sig_e):
    """get_stats_of_model for checking a smaller dt.
    Need to specify int_step which is after how many steps I store the results."""

    sim_time,h,a,rate,kramer_sig = run_sim_2D_grid_adapt_noise_smalldt(sim_len,int_step,sigmoidal_transfer,f_rhs,params_sigmoid,params_weights,*sig_e)
    results = compute_stats_helper(sim_time,rate)

    return results,kramer_sig



def get_stats_durations(sim_len,params_sigmoid,params_weights,run_sim_fct,f_rhs,*sig_e):
    """Compute for one parametersetting w and I the statistics over one trial of sim_len.
    In addition to the function get_stats_of_model this function also returns the Up and Down Durations as well as the time series for the simulation.
    This function is not optimized to run over a grid but rather to investigate results obtained for one point in the grid."""
    # plotting
    start_time = 60 # for plotting
    end_time = 80 # for plotting
    # run simulation and moving average
    sim_time,h,a,rate,kramer_sig = run_sim_fct(sim_len,sigmoidal_transfer,f_rhs,params_sigmoid,params_weights,*sig_e)
    sim_time_cut,rate_cut,mov_avg_cut = mov_avg_sim_time_rate_cut_off(rate,sim_time,100,1,0.5) # cut off first second and last 0.5 seconds

    sim_cut_results = sim_time_cut,rate_cut,mov_avg_cut
    simulation_results = sim_time,h,a,rate,kramer_sig # storage of simulation results

    every_n_point = int(len(mov_avg_cut)/7000)
    mov_avg_less_data = np.msort(mov_avg_cut)[::every_n_point]
    _,pval,intervals = dip.diptst(mov_avg_less_data)

    # compute the thresholds from the data
    threshs,bimodal = stats_UD.quartic_polynomial_fit(mov_avg_cut,50)
    print("pvalue",np.round(pval,3))
    print("variance of firing rate",np.round(np.var(rate_cut),3))
    if bimodal: # bimodal distribution, if there are thresholds found and the histogram has bigger support than 5
        print("Up and Down Transitions present!")
        threshDOWN,threshUP = threshs
        # compute crossings
        crossings,start_down,UP_dur,DOWN_dur = stats_UD.crossings_durations_2thresh(sim_time_cut,mov_avg_cut,threshUP,threshDOWN)
        if UP_dur.size != 0 or DOWN_dur.size != 0:# we want to have non empty lists for Up and Down durations, can happen when thresholds are badly chosen
            percUP,percDOWN = stats_UD.ratio_simtime_up_down(UP_dur,DOWN_dur)# compute percUp percDown
            mean_up,cv_up,mean_down,cv_down = stats_UD.mean_cv(UP_dur,DOWN_dur)# compute CV and mean UP & DOWN
            results = (percUP,percDOWN,mean_up,cv_up,mean_down,cv_down)
        else:
            print("No Up and Down transitions: No Crossings!")
            percUP,percDOWN,mean_up,cv_up,mean_down,cv_down = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
            results = (percUP,percDOWN,mean_up,cv_up,mean_down,cv_down)

        return results,simulation_results,sim_cut_results,crossings,UP_dur,DOWN_dur # return the stats but also the simulation results and Up and Down Durations
    else:
        print("No Up and Down transitions!")
        plt.plot(sim_time_cut[int(start_time/dt):int(end_time/dt)],rate_cut[int(start_time/dt):int(end_time/dt)],label = "firing rate",linewidth = 2)
        plt.plot(sim_time_cut[int(start_time/dt):int(end_time/dt)],mov_avg_cut[int(start_time/dt):int(end_time/dt)],label = "moving average",linewidth = 2)
        plt.legend()
        plt.show()
        percUP,percDOWN,mean_up,cv_up,mean_down,cv_down = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        results = (percUP,percDOWN,mean_up,cv_up,mean_down,cv_down)
        return results,simulation_results,sim_cut_results,[],[],[] # return the stats but also the simulation results and crossings,Up and Down Durations
