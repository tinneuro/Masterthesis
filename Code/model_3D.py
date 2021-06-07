# importing python modules
import numpy as np
# import own scripts
import statistics as stats_UD

#-------------------------
# This script is a cleaned up / new version of up_down_no_units.py to do analysis of the 3D model
# We already assume that we use the sigmoidal transfer function. There is no possible choice between the transfer functions anymore
# we do not store anything for Kramer-Moyal as this can then be done if needed but right now it looks like we are not going to get there
#-------------------------

r_max = 100 # maximal firing rate in Hz
dt = 0.00004 # step for Euler in seconds
N_E = 1200
N_I = 300
tau_e = 0.01
tau_i = 0.002
tau_a = 1

def sigmoidal_transfer(h,g,theta):
    """Sigmoidal transfer function."""
    phi = r_max/(1 + np.exp(-g*(h-theta)))
    return phi

def pop_activity(r,N):
    """Computing the population activity using the Poisson distribution to create some noise"""
    pois = np.random.poisson(N*r*dt)
    A = pois/(N*dt)
    return A

def f_poisson(h_e_i_a):
    """Computing the right hand side for the Euler method.
    We can decide which transfer function to use in the input."""

    h_e,h_i,a = h_e_i_a

    r_E = sigmoidal_transfer(h_e,g_e,theta_e)
    r_I = sigmoidal_transfer(h_i,g_i,theta_i)

    A_E = pop_activity(r_E,N_E)
    A_I = pop_activity(r_I,N_I)

    pop_vec = np.array([A_E,A_I])

    f_e_i_a = matrix_tau@h_e_i_a + matrix_weights@pop_vec + ext_inputs_vect
    r_e_i = [r_E,r_I]

    return f_e_i_a,r_e_i

def f_mult_gauss(h_e_i_a):
    """Computing the right hand side for the Euler method with gaussian approximation of multiplicative noise.
    Replacing the population activity with GWN so we also have noise in excitation, inhibition and adaptation here with GWN."""
    h_e,h_i,a = h_e_i_a

    r_E = sigmoidal_transfer(h_e,g_e,theta_e)
    r_I = sigmoidal_transfer(h_i,g_i,theta_i)
    sqrt_re = np.sqrt(r_E)
    sqrt_ri = np.sqrt(r_I)

    sqrtdt = np.sqrt(dt) # for the stochastic term, we multiply by dt in the euler step dt/sqrt(dt) = sqrt(dt) which is Euler-Maryuama

    xi_exc = np.random.randn()/sqrtdt # correct scaling of the random variables for Euler-Maryuama as later multiplied with dt
    xi_inh = np.random.randn()/sqrtdt

    rate_vec = np.array([r_E,r_I])
    noise_vec = np.array([sqrt_re/sqrtNE*xi_exc,sqrt_ri/sqrtNI*xi_inh])

    noise_mult_matrix = matrix_noise@noise_vec

    f_e_i_a_det = matrix_tau@h_e_i_a + matrix_weights@rate_vec + ext_inputs_vect
    f_e_i_a = f_e_i_a_det + noise_mult_matrix

    r_e_i = [r_E,r_I]

    return f_e_i_a,r_e_i


def run_sim(stop_time,f_rhs,params,params_weights):
    # define these global variables to be able to use them in the function above f_fast
    global g_e,g_i,theta_e,theta_i,I_E,I_I
    global beta,w_EE,w_EI,w_IE,w_II
    global matrix_tau,matrix_weights,ext_inputs_vect,matrix_noise
    global sqrtNE,sqrtNI

    # set the global variables to the desired values
    g_e,g_i,theta_e,theta_i,I_E,I_I = params
    beta,w_EE,w_EI,w_IE,w_II = params_weights
    sqrtNE = np.sqrt(N_E)
    sqrtNI = np.sqrt(N_I)

    # create matrices for computation of rhs Euler
    matrix_tau = np.array(([-1/tau_e,0,-1/tau_e],[0,-1/tau_i,0],[0,0,-1/tau_a])) # the rhs is always filtered by the time constant. The noise as well
    matrix_weights = np.array(([1/tau_e*w_EE,-1/tau_e*w_EI],[1/tau_i*w_IE,-1/tau_i*w_II],[beta/tau_a,0]))
    ext_inputs_vect = np.array([(1/tau_e)*I_E,(1/tau_i)*I_I,0])
    matrix_noise = np.array(([w_EE/tau_e,-w_EI/tau_e],[w_IE/tau_i,-w_II/tau_i],[beta/tau_a,0]))

    # Initialisation
    # time grid
    sim_time = np.arange(0,stop_time,dt)
    sim_length = len(sim_time)

    # storage of he,hi,a and the rates
    h_e_i_a = np.zeros((sim_length,3))
    r = np.zeros((sim_length,2))

    for i in range(sim_length-1):
    # Run the simulation using Eulers method
        # computation of rhs
        f_e_i_a,r_e_i = f_rhs(h_e_i_a[i,:])
        # Perform Euler step
        h_e_i_a[i+1,:] = h_e_i_a[i,:] + dt*f_e_i_a
        # storing the rates
        r[i+1,:] = r_e_i


    return sim_time,h_e_i_a,r



# getting results of the 3D model statistics
def get_stats_of_model_3D(sim_len,f_rhs,params_sigmoid,params_weights):
    """Compute for one parametersetting of params_weights the statistics over one trial of sim_len.
    Depending on the choice of f_sim there can be noise in the adaptation or not.
    Depending on f_rhs the multiplicative or additive noise model can be used."""

    sim_time,h_e_i_a,r = run_sim(sim_len,f_rhs,params_sigmoid,params_weights)
    rate_exc,rate_inh = r[:,0],r[:,1]
    results = compute_stats_helper_3D(sim_time,rate_exc)

    return results

def mov_avg_sim_time_rate_cut_off_3D(x,sim_time,window_avg,start_cut,end_cut):
    """Moving average + cutting of beginning and end"""
    timestep = sim_time[1] - sim_time[0] #computing dt from the timeseries. important because dt used for simulation smaller than dt used for storage
    cut_off_start = int(start_cut/timestep)
    cut_off_end = int(end_cut/timestep)
    mov_avg = stats_UD.moving_average(x,window_avg)
    mov_avg_cut = mov_avg[cut_off_start:-cut_off_end]
    sim_time_cut = sim_time[cut_off_start:-cut_off_end]
    rate_cut = x[cut_off_start:-cut_off_end]

    return sim_time_cut,rate_cut,mov_avg_cut

def compute_stats_helper_3D(sim_time,rate):
    # plotting
    start_time = 60
    end_time = 80

    sim_time_cut,rate_cut,mov_avg_cut = mov_avg_sim_time_rate_cut_off_3D(rate,sim_time,200,1,0.5)# calculate mov_avg from cut off rate and sim time
    #every_n_point = int(len(mov_avg_cut)/7000) # pvalue data reduction
    #mov_avg_less_data = np.msort(mov_avg_cut)[::every_n_point]
    #_,pval,intervals = dip.diptst(mov_avg_less_data)

    # compute the thresholds from the data
    threshs,bimodal = stats_UD.quartic_polynomial_fit(mov_avg_cut,50) # 50 is the amount of bins used
    #print("pvalue",np.round(pval,3))
    #print("variance of firing rate",np.round(np.var(rate_cut),3))
    if bimodal: # bimodal distribution, if there are thresholds found and the histogram has bigger support than 5
        print("Up and Down Transitions present!")
        threshDOWN,threshUP = threshs
        # compute crossings
        crossings,start_down,UP_dur,DOWN_dur = stats_UD.crossings_durations_2thresh(sim_time_cut,mov_avg_cut,threshUP,threshDOWN)
        if UP_dur.size != 0 or DOWN_dur.size != 0:# we want to have non empty lists for Up and Down durations, can happen when thresholds are badly chosen
            #plt_funcs.mov_avg_crossings_2thr_plot(sim_time_cut,rate_cut,mov_avg_cut,crossings,100,dt,threshUP,threshDOWN,start_time,end_time,"save_path",saveplot = False)
            percUP,percDOWN = stats_UD.ratio_simtime_up_down(UP_dur,DOWN_dur)# compute percUp percDown
            mean_up,cv_up,mean_down,cv_down = stats_UD.mean_cv(UP_dur,DOWN_dur)# compute CV and mean UP & DOWN
            if len(UP_dur) >= 400:#check that we have enough transitions for SCC!
                if len(UP_dur) == len(DOWN_dur):
                    print("Len UP/DOWN dur equal len",len(UP_dur),len(DOWN_dur))
                    serial_corr,cov = stats_UD.compute_scc_cov(0,1,UP_dur,DOWN_dur) # computing SCC for lag 0 and 1
                    scc_lag0,scc_lag1 = serial_corr # lag 0 D->U, lag1 U->D
                    cov_lag0,cov_lag1 = cov
                else:
                    print("Len UP/DOWN dur no equal len",len(UP_dur),len(DOWN_dur))
                    serial_corr,cov = stats_UD.compute_scc_cov(0,1,UP_dur,DOWN_dur[:-1])
                    scc_lag0,scc_lag1 = serial_corr
                    cov_lag0,cov_lag1 = cov
                results = (percUP,percDOWN,mean_up,cv_up,mean_down,cv_down,scc_lag0,scc_lag1,cov_lag0,cov_lag1)
            else:# to few transitions to properly calculate SCC!
                results = (percUP,percDOWN,mean_up,cv_up,mean_down,cv_down,np.nan,np.nan,np.nan,np.nan)

        else:
            print("No Up and Down transitions: No Crossings!")
            percUP,percDOWN,mean_up,cv_up,mean_down,cv_down,scc_lag0,scc_lag1,cov_lag0,cov_lag1 = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
            results = (percUP,percDOWN,mean_up,cv_up,mean_down,cv_down,scc_lag0,scc_lag1,cov_lag0,cov_lag1)

        return results
    else:
        print("No Up and Down transitions!")
        #plt.plot(sim_time_cut[int(start_time/dt):int(end_time/dt)],rate_cut[int(start_time/dt):int(end_time/dt)],label = "firing rate",linewidth = 2)
        #plt.plot(sim_time_cut[int(start_time/dt):int(end_time/dt)],mov_avg_cut[int(start_time/dt):int(end_time/dt)],label = "moving average",linewidth = 2)
        #plt.legend()
        #plt.show()
        percUP,percDOWN,mean_up,cv_up,mean_down,cv_down,scc_lag0,scc_lag1,cov_lag0,cov_lag1 = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        results = (percUP,percDOWN,mean_up,cv_up,mean_down,cv_down,scc_lag0,scc_lag1,cov_lag0,cov_lag1)
        return results

#------------Checking whether dt is small enough-------------------
def run_sim_smalldt(stop_time,integration_step,f_rhs,params,params_weights):
    # define these global variables to be able to use them in the function above f_fast
    global g_e,g_i,theta_e,theta_i,I_E,I_I
    global beta,w_EE,w_EI,w_IE,w_II
    global matrix_tau,matrix_weights,ext_inputs_vect,matrix_noise
    global sqrtNE,sqrtNI

    # set the global variables to the desired values
    g_e,g_i,theta_e,theta_i,I_E,I_I = params
    beta,w_EE,w_EI,w_IE,w_II = params_weights
    sqrtNE = np.sqrt(N_E)
    sqrtNI = np.sqrt(N_I)

    # create matrices for computation of rhs Euler
    matrix_tau = np.array(([-1/tau_e,0,-1/tau_e],[0,-1/tau_i,0],[0,0,-1/tau_a])) # the rhs is always filtered by the time constant. The noise as well
    matrix_weights = np.array(([1/tau_e*w_EE,-1/tau_e*w_EI],[1/tau_i*w_IE,-1/tau_i*w_II],[beta/tau_a,0]))
    ext_inputs_vect = np.array([(1/tau_e)*I_E,(1/tau_i)*I_I,0])
    matrix_noise = np.array(([w_EE/tau_e,-w_EI/tau_e],[w_IE/tau_i,-w_II/tau_i],[beta/tau_a,0]))

    # Initialisation
    # time grid
    sim_time = np.arange(0,stop_time,dt)
    sim_length = len(sim_time)

    # storage of he,hi,a and the rates
    storage_length = int(sim_length/integration_step)
    # storage of he,a and the rate
    sim_time_store = np.arange(0,stop_time,integration_step*dt)

    h_e_i_a = np.zeros((storage_length,3))
    r = np.zeros((storage_length,2))

    h_eia_prev = np.array([0,0,0])#initialisation

    for i in range(sim_length-1):
    # Run the simulation using Eulers method
        # computation of rhs
        f_e_i_a,r_e_i = f_rhs(h_eia_prev)
        # Perform Euler step
        h_eia_next = h_eia_prev + dt*f_e_i_a

        h_eia_prev = h_eia_next

        # store the result every integration_step but not for the first. First one stays zero
        if (i % integration_step == 0) & (i != 0):
            j = int(i/integration_step) # starts with 1 for i = integration_step
            h_e_i_a[j] = h_eia_next

            r[j] = r_e_i


    return sim_time_store,h_e_i_a,r
