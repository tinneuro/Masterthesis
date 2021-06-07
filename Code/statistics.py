# python modules
import numpy as np
import math
from scipy.fft import fft
from scipy.optimize import curve_fit
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import cycler

# own scripts
#import up_down_no_units as ups_fast #scripts for functions which are commented out and currently not needed
#import several_trials as trials


# colors from ggplot
colors = cycler('color',
                ['#E24A33','#348ABD','#988ED5','#988ED5','#777777','#FBC15E','#8EBA42','#FFB5B8'])

def moving_average(x,window_avg):
    """Calculate a moving average of the excitatory firing rate"""
    # window_avg is a number how many points in time we use for averaging the signal
    # with setting same the timings are correct wherease with the setting valid the signal is shifted to the left
    return np.convolve(x, np.ones(window_avg)/window_avg, mode='same')

def mov_avg_sim_time_rate_cut_off(x,sim_time,window_avg,start_cut,end_cut):
   """Moving average + cutting of beginning and end"""
   timestep = sim_time[1] - sim_time[0] #compute dt from data
   cut_off_start = int(start_cut/timestep)
   cut_off_end = int(end_cut/timestep)
   mov_avg = moving_average(x,window_avg)
   mov_avg_cut = mov_avg[cut_off_start:-cut_off_end]
   sim_time_cut = sim_time[cut_off_start:-cut_off_end]
   rate_cut = x[cut_off_start:-cut_off_end]

   return sim_time_cut,rate_cut,mov_avg_cut

def quartic_poly(x,a,b,c,d,e):
    """Quartic polynomial with coefficients a,b,c,d,e"""
    return a*x**4 + b*x**3 + c*x**2  + d*x + e

def find_inflection_points(coeffs):
    """Given the coefficients of the quartic polynomial fit we find the inflection points.
    If there are no inflection points because we have a negative root then an empty list and False is given back."""
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]

    p_half = b/(4*a)
    D = p_half**2 - c/(6*a)
    if D>0:
        root_D = np.sqrt(D)
        x1 = -p_half + root_D
        x2 = -p_half - root_D
        return [x1,x2],True
    else:
        print("0 inflection points")
        return [],False

def check_bimodality(infl_points,max_bin,min_bin):
    """Checking whether we have Up and Down transitions.
    Condition: Both inflection points need to lie within the support of the histogram and the support needs to be >= 5.
    bimodal = True says that we have U,D transitions
    bimodal = False no U,D transitions."""

    infl_point_1,infl_point_2 = infl_points

    if (max_bin > infl_point_1 >= min_bin) & (max_bin > infl_point_2 >= min_bin) & (max_bin - min_bin >=5):# adding condition check support of hist
        bimodal = True
    else:
        bimodal = False

    return bimodal

def quartic_polynomial_fit(mov_avg,bins,save_path = None,saveplot = False):
    """Based on the moving average and the number of bins we compute the thresholds for U and D classification.
    We use a quartic polynomial to fit to the logarithmic histogram of the moving average histogram.
    Classification whether U,D transitions are present or not is included."""
    plt.style.use('default')
    params = {'figure.figsize': (10,6),
            'axes.prop_cycle': colors,
              'lines.linewidth': 3,
              'legend.fontsize': 20,
             'axes.labelsize': 20,
             'axes.titlesize':30,
             'xtick.labelsize':15,
             'ytick.labelsize':15,
              'xtick.major.size': 10,
               'xtick.major.width' : 2,
               'xtick.minor.size' :5,
               'xtick.minor.width' : 1,
              'ytick.major.size': 10,
               'ytick.major.width' : 2,
               'ytick.minor.size' :5,
               'ytick.minor.width' : 2,
             'figure.constrained_layout.use': True}
    plt.rcParams.update(params)#plot_params

    hist,bins = np.histogram(mov_avg,bins= bins)
    max_bin,min_bin = np.max(bins),np.min(bins)
    hist[np.where(hist ==0)] = 1 # intermediate values where there is no data set to one otherwise get a -inf which we are not able to cope with
    log_hist = np.log(hist)
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    coeffs, pcov = curve_fit(quartic_poly, binscenters, log_hist, bounds=(-np.inf, [-0.01, np.inf,np.inf,np.inf,np.inf])) #fit a quartic polynomial with constraints
    #print("coefficients poly",coeffs)
    infl_points, boolean = find_inflection_points(coeffs)

    a,b,c,d,e = coeffs
    poly_fit = quartic_poly(binscenters,a,b,c,d,e)

    x_big =  np.linspace(-1000,3000,100)
    poly_fit_2 = quartic_poly(x_big,a,b,c,d,e)
    #plt.title("big picture of quartic fit")

    #plt.plot(x_big,poly_fit_2,label = "quartic polynomial fit",color = 'darkorange')
    #plt.plot(binscenters,log_hist,label = "logarithmic histogram of moving average",color = '#348ABD',linewidth = 5)

    #plt.legend(fontsize = 25,loc = 'center left',bbox_to_anchor=(1, 0.5))
    #plt.show()

    #plt.title("Quartic fit of loghist")
    #plt.plot(binscenters,log_hist,label = "Logarithmic histogram of moving average",color = '#348ABD')
    #plt.plot(binscenters,poly_fit,label = "Quartic polynomial fit",color = 'darkorange')
    #plt.xlabel("Rate in Hz")
    #plt.ylabel("Log Count")

    if boolean: # True when there were two inflection points
        bimodal = check_bimodality(infl_points,max_bin,min_bin)
        print("infl points",infl_points)
        if bimodal:
            threshs = np.sort(infl_points)
            threshDOWN = np.round(threshs[0],2)
            threshUP = np.round(threshs[1],2)
            #plt.plot([threshDOWN,threshDOWN],[0,max(log_hist)],label = "Thresholds",color = "black")
            #plt.plot([threshUP,threshUP],[0,max(log_hist)],color = "black")
            #plt.legend(loc = 'lower center',bbox_to_anchor=(0.5, -0.5))
            #if saveplot:
                #plt.savefig(save_path,dpi=200)

            #plt.show()
            print("THRESH DOWN",threshDOWN)
            print("THRESH UP",threshUP)

            return [threshDOWN,threshUP],True

        else:
            print("No Up and Down Dynamics: Inflection point outside support")
            #plt.legend(fontsize = 25,loc = 'center left',bbox_to_anchor=(1, 0.5))
            #plt.show()

            return [], False
    else:
        print("No Up and Down Dynamics: No inflection points")
        #plt.legend(fontsize = 25,loc = 'center left',bbox_to_anchor=(1, 0.5))
        #plt.show()

        return [], False

def quartic_polynomial_fit_nologhist(mov_avg,bins):
    """Checking what happens for Threshold Classification when not the logarithmic histogram is used."""
    hist,bins = np.histogram(mov_avg,bins= bins)
    max_bin,min_bin = np.max(bins),np.min(bins)
    #hist[np.where(hist ==0)] = 1 # intermediate values where there is no data set to one otherwise get a -inf which we are not able to cope with
    #log_hist = np.log(hist)
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    coeffs, pcov = curve_fit(quartic_poly, binscenters, hist, bounds=(-np.inf, [-0.01, np.inf,np.inf,np.inf,np.inf])) #fit a quartic polynomial with constraints
    #print("coefficients poly",coeffs)
    infl_points, boolean = find_inflection_points(coeffs)

    a,b,c,d,e = coeffs
    poly_fit = quartic_poly(binscenters,a,b,c,d,e)

    x_big =  np.linspace(-1000,3000,100)
    poly_fit_2 = quartic_poly(x_big,a,b,c,d,e)
    plt.title("big picture of quartic fit")
    plt.plot(binscenters,hist,label = "hist",linewidth = 3)
    plt.plot(x_big,poly_fit_2,label = "quartic fit",linewidth = 3)

    #plt.legend(fontsize = 25,loc = 'center left',bbox_to_anchor=(1, 0.5))
    plt.show()

    plt.title("Quartic fit of loghist")
    plt.plot(binscenters,hist,label = "hist",linewidth = 3)
    plt.plot(binscenters,poly_fit,label = "quartic fit",linewidth = 3)

    if boolean: # True when there were two inflection points
        bimodal = check_bimodality(infl_points,max_bin,min_bin)
        if bimodal:
            threshs = np.sort(infl_points)
            threshDOWN = np.round(threshs[0],2)
            threshUP = np.round(threshs[1],2)
            plt.plot([threshDOWN,threshDOWN],[0,max(hist)],label = "Threshold Down",linewidth = 3)
            plt.plot([threshUP,threshUP],[0,max(hist)],label = "Threshold Up",linewidth = 3)
            #plt.legend(fontsize = 25,loc = 'center left',bbox_to_anchor=(1, 0.5))
            plt.show()
            print("THRESH DOWN",threshDOWN)
            print("THRESH UP",threshUP)

            return [threshDOWN,threshUP],True

        else:
            print("No Up and Down Dynamics: Inflection point outside support")
            plt.legend(fontsize = 25,loc = 'center left',bbox_to_anchor=(1, 0.5))
            plt.show()

            return [], False
    else:
        print("No Up and Down Dynamics: No inflection points")
        #plt.legend(fontsize = 25,loc = 'center left',bbox_to_anchor=(1, 0.5))
        plt.show()

        return [], False


def detect_transitions(x,threshold_up):
    """Detection of transitions between U and D using only one threshold.
    x is the time series of the moving average firing rate.
    threshold_up is the threshold where we classfiy everything above as as Up and below as Down
    Returns indices where the crossing below -> above and above -> below threshold happened
    and whether we start in Down or Up State."""
    # create a variable which tells whether we start in Down or Up State at the beginning
    start_down = True

    diff = x - threshold_up
    if diff[0] >= 0:
        start_down = False

    # calculate the crossing indices
    crossing_idx = np.where(np.diff(np.signbit(diff)))

    return crossing_idx[0],start_down

def detect_transitions_2thresholds(x,down_to_up_thr,up_to_down_thr):
    """Detection of transitions across two thresholds. We have differing thresholds for U->D transition which will
     be lower than the threshold for D->U Up transistions. This is done to avoid spourious detections of U,D due to noise.
    x is the time series of the moving average firing rate
    down_to_up_thr is the threshold for Down -> Up crossings
    up_to_down_thr is the threshold for Up -> Down crossings
    Return: two arrays containing ALL Up->Down transitions and the other ALL Down->Up transitions.
    These still need to be processed by another function to clear out those transitions caused by noise."""

    diff_DU = x - down_to_up_thr # Down to Up transitions (- to +)
    diff_UD = x - up_to_down_thr # Up to Down transitions (+ to -)

    # Crossings U -> D
    sign_UD = np.sign(diff_UD)
    sign_UD[sign_UD == 0] = 1 # the 0 entries get 0 as sign and make problems
    crossing_UD = np.where(np.diff(sign_UD) < 0)

    # Crossings D -> U
    sign_DU = np.sign(diff_DU)
    sign_DU[sign_DU == 0] = 1 # the 0 entries get 0 as sign and make problems
    crossing_DU = np.where(np.diff(sign_DU) > 0)


    return crossing_UD[0],crossing_DU[0]


def get_crossings_2thresh(crossing_DU,crossing_UD):
    """This gets one crossings list out of the two crossing lists for D->U and U->D crossings.
    The algorithm calculates the real crossings as there can be crossings of the Up boundary just because of fluctuations but it should still be one up state.
    We obtain a final list of the real crossings and discard all those which are crossings via fluctuations.
    The algorithm automatically only starts from the Down state before the second Up state! Everything before will not be considered.
    It will use the last UD transition to find where the proper crossing from DU is for this second Up state.
    So we most of the time lose the first Up and Down. This is especially bad for time series where there are very long states.
    This algorithm will always start in an Down the way it is constructed. (see if i == 0:
    crossings.insert(0,last_UD))
    Sequence will be D,U,D,U,D,...
    We will end with the last Up state. No Down states after are considered.

    Input: the two arrays from the function detect_transitions_2thresholds
    Return: a list of crossings, where the first two crossings correspond to Up state, (first crossing is D->U)
    a boolean variable start_down which will always be set to True as we will always start with an Up state"""
    crossings = []
    first = True #first iteration

    for i in range(len(crossing_UD)-1):
        next_UD = crossing_UD[i+1] # store the next UD crossing
        last_UD = crossing_UD[i] # store last UD crossing

        crossing_DU_between = crossing_DU[np.logical_and(crossing_DU>=last_UD, crossing_DU<=next_UD)] # get all DU crossings which are in between them

        if len(crossing_DU_between) == 1: # if there was only one DU crossing in between take it, easiest case
            crossings.append(crossing_DU_between[0])
            crossings.append(next_UD)
            if first: # for the first round also append the Down state which is in between the Ups
                crossings.insert(0,last_UD)# insert the very first Down as well.
                first = False

            if (i == len(crossing_UD)-2) and (crossing_DU[-1] > crossing_UD[-1]):# caring about the last Down state
            # and if we have a very last DU but no UD anymore (end in an Up which is not done)
                crossing_DU_end = crossing_DU[crossing_DU>= crossing_UD[-1]] # need to chose the first DU crossing again there might be several still in the back
                crossings.append(np.min(crossing_DU_end)) # append the last crossing towards an Up to also have the last Down

        elif len(crossing_DU_between) > 1: # if there are several DU crossings in between take the smallest one (first one and not the intermediate ones)
            crossings.append(np.min(crossing_DU_between[0]))
            crossings.append(next_UD)
            if first: # for the first round also append the Down state which is in between the Ups
                crossings.insert(0,last_UD)# insert the very first Down as well.
                first = False

            if (i == len(crossing_UD)-2) and (crossing_DU[-1] > crossing_UD[-1]): # caring about the last Down state
            # and if we have a very last DU but no UD anymore (end in an Up which is not done)
                crossing_DU_end = crossing_DU[crossing_DU>= crossing_UD[-1]]# need to chose the first DU crossing again there might be several still in the back
                crossings.append(np.min(crossing_DU_end)) # append the last crossing towards an Up to also have the last Down

        # if there are no DU crossings in between then do not classify as Up state, therefore do not add anything to crossings

    start_down = False # The way this algorithm is constructed we always start with an Down state (if first)
    crossings = np.array(crossings)

    return crossings,start_down

def find_up_down_durations(time,crossing_indices,start_down,first_state = False):
    """Calculate the Up and Down State durations from the time, the crossing indices and whether we start in Up or Down.
    The first Down state (initial condition -> first Up) can be included setting first_state = True.
    If the first state is not included then the ordering will be UP -> Down if we started with down(which is not included).
    If we start with (0,0,0) then we start in the down state.
    Return: array of Up and array of Down durations"""
    # the last bit of the dynamics where either the Up State or the Down State is not ready is not included.
    # Therefore it is possible to have not equal amounts of Ups and Down states in one simulation.
    # Should we include the first state?
    # But this might be useful like this because we do not know anyways how long this Up or Down State would have been!

    crossing_times = time[crossing_indices]
    durations_ups_downs = np.diff(crossing_times)
    if start_down:
        down_1 = crossing_times[0] #up until the first crossing we are in a Down State, not using it
        Ups = durations_ups_downs[::2] # take the 1st,3rd,5th,... element
        Downs = durations_ups_downs[1::2]# take the 2nd,4th,6th element

        U_list = Ups
        D_list = Downs

        # add the first down state of simulation as well
        # Is this really produced by the dynamics?
        if first_state:
            D_list = np.insert(D_list,0,down_1)

    else:
        # if we start in Up State then the first element in durations_ups_downs is the duration of a Down State
        up_1 = crossing_times[0] #up until the first crossing we are in a Up State, not using it
        Downs = durations_ups_downs[::2]
        Ups = durations_ups_downs[1::2]

        U_list = Ups
        D_list = Downs
        # add the first down state of simulation as well
        # Is this really produced by the dynamics?
        if first_state:
            U_list = np.insert(U_list,0,up_1)


    return U_list,D_list

def mov_avg_crossings_durations(sim_time,rate,window_avg,threshold_up):
    """Apply the three functions from above to calculate moving average, crossings and Up and Down durations"""
    mov_avg = moving_average(rate,window_avg)
    crossings,start_down = detect_transitions(mov_avg,threshold_up)
    UP_dur, DOWN_dur = find_up_down_durations(sim_time,crossings,start_down)

    return mov_avg,crossings,start_down,UP_dur,DOWN_dur


def mov_avg_crossings_durations_2thresholds(sim_time,rate,window_avg,down_to_up_thr,up_to_down_thr):
    """Same function as mov_avg_crossings_durations just for 2 thresholds for crossings
    We need to apply one more function to be able to caculate the crossings compared to above with only one threshold."""

    mov_avg = moving_average(rate,window_avg)
    crossing_UD,crossing_DU = detect_transitions_2thresholds(mov_avg,down_to_up_thr,up_to_down_thr)
    crossings,start_down = get_crossings_2thresh(crossing_DU,crossing_UD)

    if crossings.size != 0: # check if there were crossings above this threshold at all
        UP_dur, DOWN_dur = find_up_down_durations(sim_time,crossings,start_down)
        return mov_avg,crossings,start_down,UP_dur,DOWN_dur

    else:
        print("No Crossings found")
        return mov_avg,crossings,start_down,np.array([]),np.array([])

def crossings_durations_2thresh(sim_time,mov_avg,down_to_up_thr,up_to_down_thr):
    """Same function as above but without calculating the moving average because already done this to calculate the thresholds"""

    crossing_UD,crossing_DU = detect_transitions_2thresholds(mov_avg,down_to_up_thr,up_to_down_thr)
    crossings,start_down = get_crossings_2thresh(crossing_DU,crossing_UD)
    if crossings.size != 0:# check if there were crossings above this threshold at all
        UP_dur, DOWN_dur = find_up_down_durations(sim_time,crossings,start_down)
        return crossings,start_down,UP_dur,DOWN_dur
    else:
        print("No Crossings found")
        return crossings,start_down,np.array([]),np.array([])



def find_thresholds(mov_avg):
    """Find the Up, inter-peak trough and Down peak in the data.
    Old algorithm which uses Kernel Density estimation. (KDE)
    Chosing the inter-peak-trough to be the middle of the support of the moving average histogram. Chosing the peaks
    as maxima of the KDE of left and right histograms. No need for analytical maxima and minima anymore.
    To not make this function take too much time we need to reduce the time series of the rate data before, e.g. only take every 100th element."""

    # plotting params
    plt.style.use('default')
    params = {'figure.figsize': (10,6),
             'axes.prop_cycle': colors,
              'lines.linewidth': 1.5,
              'legend.fontsize': 'x-large',
             'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large',
             'figure.constrained_layout.use': False}
    plt.rcParams.update(params)
    # cutting histogram into halfs based on the middle of the support
    rate_middle = (np.max(mov_avg) + np.min(mov_avg))/2

    left_hist = mov_avg[mov_avg < rate_middle]
    right_hist = mov_avg[mov_avg >= rate_middle]

    kde_left = sm.nonparametric.KDEUnivariate(left_hist) #comparison KDE with histogram needed to see really necessary
    kde_left.fit(bw = 0.3)# take bigger bandwidth to rather oversmooth
    max_ind_left = np.argmax(kde_left.density)
    rate_down = kde_left.support[max_ind_left]

    kde_right = sm.nonparametric.KDEUnivariate(right_hist) # why I take kernel density estimation especially visible in the right histogram which is quite flat
    kde_right.fit(bw = 0.3)# because less data/smaller xrange I take a smaller bw. bigger bw would shift the density estimation more to the right
    max_ind_right = np.argmax(kde_right.density)
    rate_up = kde_right.support[max_ind_right]

    # calculate the thresholds from the maxima and inter-peak trough of the rate
    threshUP = (rate_middle-rate_up)*0.5 + rate_up
    threshDOWN = (rate_middle-rate_down)*0.5 + rate_down

    # rounding
    rate_up = np.round(rate_up,2)
    rate_down = np.round(rate_down,2)
    rate_middle = np.round(rate_middle,2)

    threshUP = np.round(threshUP,2)
    threshDOWN = np.round(threshDOWN,2)

    print("rate Down",rate_down)
    print("rate middle",rate_middle)
    print("rate Up",rate_up)

    print("THRESH UP",threshUP)
    print("THRESH DOWN",threshDOWN)

    plt.plot(kde_left.support,kde_left.density,linewidth = 3,label = "KDE left")
    plt.hist(left_hist,density = True,bins =30,label = "Histogram of moving average left")
    plt.plot([rate_down,rate_down],[0,1],linewidth = 3,color = "green",label = f"rate down: {rate_down}")
    plt.legend(loc = "upper right")
    plt.show()

    plt.plot(kde_right.support,kde_right.density,linewidth = 3,label = "KDE right")
    plt.hist(right_hist,density = True,bins =30,label = "Histogram of moving average right")
    plt.plot([rate_up,rate_up],[0,1],linewidth = 3,color = "green",label = f"rate up: {rate_up}")
    plt.legend(loc = "upper right")
    plt.show()

    # Plotting of thresholds
    plt.figure(figsize = (16,10))
    plt.hist(mov_avg,density = True,bins =30,label = "Histogram of moving average")
    plt.plot([threshUP,threshUP],[0,1],linewidth = 3,color = "black",linestyle = "dashed",label = f"threshUP: {threshUP}")
    plt.plot([threshDOWN,threshDOWN],[0,1],linewidth = 3,color = "black",label = f"threshDOWN: {threshDOWN}")
    plt.plot([rate_down,rate_down],[0,1],linewidth = 3,color = "#ACDF87",label = f"rate down: {rate_down}")
    plt.plot([rate_middle,rate_middle],[0,1],linewidth = 3,color = "#4C9A2A",label = f"rate middle: {rate_middle}")
    plt.plot([rate_up,rate_up],[0,1],linewidth = 3,color = "#1E5631",label = f"rate up: {rate_up}")
    plt.legend(loc = "upper right")
    plt.yscale('log', nonposy='clip')
    plt.show()

    return rate_up,rate_down,rate_middle,threshUP,threshDOWN

def mean_cv(up_dur,down_dur):
    """Calculate the mean and the CV of the durations"""
    UPdur_array = np.array(up_dur)
    DOWNdur_array = np.array(down_dur)

    mean_up = np.mean(UPdur_array)
    std_up = np.std(UPdur_array)
    cv_up = std_up/mean_up

    mean_down = np.mean(DOWNdur_array)
    std_down = np.std(DOWNdur_array)
    cv_down = std_down/mean_down

    return mean_up,cv_up,mean_down,cv_down

def mean_cv_skew(up_dur,down_dur):
    """Calculate the mean, CV and skew of the durations"""
    UPdur_array = np.array(up_dur)
    DOWNdur_array = np.array(down_dur)

    mean_up = np.mean(UPdur_array)
    std_up = np.std(UPdur_array)
    mean_third_up = np.mean(UPdur_array**3)

    cv_up = std_up/mean_up
    skew_up = (mean_third_up-3*mean_up*(std_up**2)-mean_up**3)/(std_up**3)

    mean_down = np.mean(DOWNdur_array)
    std_down = np.std(DOWNdur_array)
    mean_third_down = np.mean(DOWNdur_array**3)

    cv_down = std_down/mean_down
    skew_down = (mean_third_down-3*mean_down*(std_down**2)-mean_down**3)/(std_down**3)

    return mean_up,cv_up,mean_down,cv_down,skew_up,skew_down

def ratio_simtime_up_down(up_dur,down_dur):
    """Compute ratio of simulation time spent in Up or Down state
    For the denominator we consider the up + down durations and not the whole simulation time because parts of the simulation time are not included."""

    time_in_up = np.sum(up_dur)
    time_in_down = np.sum(down_dur)
    total_time = time_in_up + time_in_down

    perc_up = np.round(time_in_up/total_time*100,2)
    perc_down = np.round(time_in_down/total_time*100,2)

    return perc_up,perc_down

def SCC(U,D,lag):
    """
    returns vector of length lag. lag is a number so we return the correlation (float) for this lag
    of serial correlation coefficients of stationary sequence X
    according to definition in Jercog et al
    We assume for this function that we start in Down. D_i,U_i,D_i+1...
    lag <= 0 refers to D before U
    In particular lag = 0 refers to the immediately previous DOWN.
    lag = 1 refers to the immediately consecutive Down.
    lag > 0 refers to DOWN after UP
    """

    L=len(U)
    if lag<0:
        X=D
        Y=U
        lag= -lag
    else:
        X=U
        Y=D

    m1=np.mean(X[:L-lag])
    m2=np.mean(Y[lag:])
    #print("mean Down/Up",m1)
    #print("mean Down/Up",m2)
    s1=np.std(X[:L-lag])
    s2=np.std(Y[lag:])
    #print("standard dev s1",s1)
    #print("standard dev s2",s2)
    C=np.mean(X[:L-lag]*Y[lag:])-m1*m2

    return C/(s1*s2)

def SCC_and_COV(U,D,lag):
    """
    Same as SCC function but it returns the covariance as well not only the SCC.
    SCC is the normalized covariance.
    Return: Corr, Cov
    """

    L=len(U)
    if lag<0:
        X=D
        Y=U
        lag= -lag
    else:
        X=U
        Y=D

    m1=np.mean(X[:L-lag])
    m2=np.mean(Y[lag:])
    #print("mean Down/Up",m1)
    #print("mean Down/Up",m2)
    s1=np.std(X[:L-lag])
    s2=np.std(Y[lag:])
    #print("standard dev s1",s1)
    #print("standard dev s2",s2)
    C=np.mean(X[:L-lag]*Y[lag:])-m1*m2

    return C/(s1*s2),C

def compute_scc(start_lag,end_lag,up_dur,down_dur):
    """Compute SCC for several different lags. Return and array with the SCC for the different lags."""
    # up_dur,down_dur need to be such that we have D,U,D,U (start with down) because lag = 0 refers to immediately previous down
    lags=np.arange(start_lag,end_lag+1,1)
    rho=np.zeros(len(lags))
    for (i,lag) in enumerate(lags):
        rho[i]=SCC(up_dur,down_dur,lag)

    return rho

def compute_scc_cov(start_lag,end_lag,up_dur,down_dur):
    """Same function as compute_scc but in addition to the SCC we also give back the covariance."""
    # up_dur,down_dur need to be such that we have D,U,D,U (start with down) because lag = 0 refers to immediately previous down
    lags=np.arange(start_lag,end_lag+1,1)
    rho = np.zeros(len(lags))
    cov = np.zeros(len(lags))
    for (i,lag) in enumerate(lags):
        rho[i],cov[i] = SCC_and_COV(up_dur,down_dur,lag)

    return rho,cov


def stderror_on_mean(data):
    """Compute standard error on the mean of statistic over several trials stored in data"""
    mean = np.mean(data)
    std_error = np.std(data)/np.sqrt(len(data))

    return mean,std_error

#def run_several_trials_get_durations(stop_time,n_trials,threshold,window_length,params,params_weights):
#    """Run the trials several times and store the Up and Down durations. Multiplicative noise case.
#    Input: stop_time = length of one trial in seconds
#            n_trials = amount of trials
#            threshold = threshold for Up Down detections
#            window_length = lenght of window for the average in moving_average
#            """
#
#    # total up/down durations over the trials
#    up_dur_total = []
#    down_dur_total = []
#
#    # having seperate lists for the different runs to store them
#    up_dur_trials = []
#    down_dur_trials = []
#    start_down_trials = []
#
#    # count how many Ups and Downs we observed
#    up_counts_trials = []
#    down_counts_trials = []
#
#    for i in range(n_trials):
#        time_sim,h_e_i_a,r = ups_fast.run_simulation_Up_Down_fast(stop_time,ups_fast.sigmoidal_transfer,ups_fast.sigmoidal_transfer,params,params_weights)
#        rate_exc = r[:,0]
#        mov_avg,crossings,start_down,UP_dur,DOWN_dur = mov_avg_crossings_durations(time_sim,rate_exc,window_length,threshold)
#        # append is bad for memory and takes a lot of time to call it if there is already a lot stored inside. tried it out should be fine
#        up_counts_trials.append(len(UP_dur))
#        down_counts_trials.append(len(DOWN_dur))
#
#        up_dur_trials.append(UP_dur)
#        down_dur_trials.append(DOWN_dur)
#        start_down_trials.append(start_down)
#
#        up_dur_total.extend(UP_dur)
#        down_dur_total.extend(DOWN_dur)
#
#    return up_dur_trials,down_dur_trials,up_counts_trials,down_counts_trials,start_down_trials,up_dur_total,down_dur_total
#
#
#def run_several_trials_get_durations_add_GWN(stop_time,n_trials,threshold,window_length,params,params_weights,sigma_exc,sigma_inh):
#    """Run the trials several times and store the Up and Down durations. Multiplicative noise case.
#            Input: stop_time = length of one trial in seconds
#            n_trials = amount of trials
#            threshold = threshold for Up Down detections
#            window_length = lenght of window for the average in moving_average
#            sigma_exc = sqrt(r_e/(N_E*dt)) matching noise in Up state
#            sigma_inh = sqrt(r_i/(N_I*dt)) matching noise in Down state
#            """
#
#    up_dur_total = np.array([])
#    down_dur_total = np.array([])
#
#    up_dur_trials = []
#    down_dur_trials = []
#    start_states_trials = []
#
#    up_counts_trials = []
#    down_counts_trials = []
#
#    for i in range(n_trials):
#        time,h_e_i_a,r = ups.run_simulation_Up_Down_fast_additive_GWN(stop_time,ups_fast.sigmoidal_transfer,ups_fast.sigmoidal_transfer,params,params_weights,sigma_exc,sigma_inh)
#        moving_avg = moving_average(r[:,0],window_length)
#        crossings,start_state = detect_transitions(moving_avg,threshold)
#        UP_dur, DOWN_dur = find_up_down_durations(time,crossings,start_state)
#        up_counts_trials.append(len(UP_dur))
#        down_counts_trials.append(len(DOWN_dur))
#
#        up_dur_trials.append(UP_dur)
#        down_dur_trials.append(DOWN_dur)
#        start_states_trials.append(start_state)
#        up_dur_total = np.concatenate((up_dur_total,UP_dur))
#        down_dur_total = np.concatenate((down_dur_total,DOWN_dur))
#
#    return up_dur_trials,down_dur_trials,up_counts_trials,down_counts_trials,start_states_trials,up_dur_total,down_dur_total


def serial_corr_multiple_trials(start_lag,end_lag,up_dur_trials,down_dur_trials,start_down_trials):
    """Compute the serial correlations for each trial and store the results in a matrix to be able to do a boxplot later"""

    n_trials = len(up_dur_trials)

    # each row has the serial correlations for the next trial, the column correspond to the different time lags
    serial_corr_trials = np.zeros((n_trials,end_lag -  start_lag + 1))

    for i in range(n_trials):
        # access the up_durations and down_durations for each trial
        UP_dur = up_dur_trials[i]
        DOWN_dur = down_dur_trials[i]
        # check whether we are first in Up or Down
        if start_down_trials[i]:
            # start in Down but not classified as Down: first state will be an Up state, so need to remove it for serial correlation analysis
            UP_dur_minus_first = UP_dur[1:]
            if len(UP_dur_minus_first) == len(DOWN_dur): # check whether we now have the same amount of Us and Ds, most of the time we are in this case (if we end in down)
                rho = compute_scc(start_lag,end_lag,UP_dur_minus_first,DOWN_dur)
                serial_corr_trials[i,:] = rho
            else: # if we end in the middle of an Up state, then the last Down state does not interest us anymore
                DOWN_dur_minus_last = DOWN_dur[:-1]
                rho = compute_scc(start_lag,end_lag,UP_dur_minus_first,DOWN_dur_minus_last)
                serial_corr_trials[i,:] = rho


        else: # start in Up state, first state will be a Down state, no need to remove first state
            if len(UP_dur) == len(DOWN_dur): # check whether we now have the same amount of Us and Ds
                rho = compute_scc(start_lag,end_lag,UP_dur,DOWN_dur)
                serial_corr_trials[i,:] = rho
            else: #if we end in Up state need to remove last down state
                DOWN_dur_minus_last = DOWN_dur[:-1]
                rho = compute_scc(start_lag,end_lag,UP_dur,DOWN_dur_minus_last)
                serial_corr_trials[i,:] = rho

    return serial_corr_trials




#def first_order_stats_from_rates_trials(sim_time,re_trials,ri_trials,mov_avg_window):
#    """This function calculates all relevant first order statistics based on the excitatory and inhibitory firing rate.
#    Furthermore to calculate the moving average we need the mov_avg_window. For finding out the thresholds for Up and Down transitions we
#    need to apply the find_thresholds function which needs the input moving average.
#    To calculate proper statistics we will remove the 1s from the simulated data as there is an intrinsic dependence on the initial condition.
#    Furthermore for the moving average we will remove the last 200 timesteps to not get the wrong behaviour at the end."""
#
#    # cutting of 1s from re and ri and 200 time steps from the back of moving average
#    cut_off_start = int(1/ups_fast.dt)
#    cut_off_end = 200
#
#    # amount of trials
#    ntrials = len(re_trials)
#
#    # conversion to array
#    re_trials_array = np.array(re_trials) # still need whole time series for computation of mov_avg
#    ri_trials_array = np.array(ri_trials)
#    re_trials_array_cut = re_trials_array[:,cut_off_start:]
##    ri_trials_array_cut = ri_trials_array[:,cut_off_start:]
#
#    # calculation of basic statistics independent of the up and down durations and moving average
#    # mean
##    re_trials_mean = np.mean(re_trials_array_cut,axis = 1)
#    ri_trials_mean = np.mean(ri_trials_array_cut,axis = 1)
#
#    re_mean = np.mean(re_trials_mean)
#    ri_mean = np.mean(ri_trials_mean)
#
#    re_mean_stderror = np.std(re_trials_mean)/np.sqrt(ntrials)
#    ri_mean_stderror = np.std(ri_trials_mean)/np.sqrt(ntrials)
#
#    # variance
#    re_trials_var = np.var(re_trials_array_cut,axis = 1)
#    ri_trials_var = np.var(ri_trials_array_cut,axis = 1)
#
#    re_var = np.mean(re_trials_var) # take average of variance over the trials
##    ri_var = np.mean(ri_trials_var)

##    re_var_stderror = np.std(re_trials_var)/np.sqrt(ntrials)
##    ri_var_stderror = np.std(ri_trials_var)/np.sqrt(ntrials)

#    # moving average calculation based on whole time series
##    mov_avg_trials = trials.moving_avg_trials(re_trials_array,mov_avg_window)
#    mov_avg_trials_cut = mov_avg_trials[:,cut_off_start:-cut_off_end] # take away the 1s and the last 200 seconds of moving average
#    sim_time_cut = sim_time[cut_off_start:-cut_off_end]
#
#    # create less data for threshold determination
#    mov_avg_trials_flat = mov_avg_trials_cut.flatten()
#    no_for_less_data = np.max((1,np.math.ceil(len(mov_avg_trials_flat)/10000000)))# want to take roughly 10mio datapoints, if we have too few data points (less than 10 mio) take every datapoint.
#    mov_avg_less_data = np.msort(mov_avg_trials_flat)[::no_for_less_data] # every "no_for_less_data" we take a data point from moving avg
#
#    rate_up,rate_down,rate_middle,threshUP,threshDOWN = find_thresholds(mov_avg_less_data)
#
    # Calculation of the UP and DOWN durations, do not need to calculate moving average as it is already done before
##    up_dur_trials,down_dur_trials,up_counts_trials,down_counts_trials,start_down_trials,up_dur_total,down_dur_total = trials.calculate_durations_from_several_trials_2thresholds_no_mov_avg(sim_time_cut,mov_avg_trials_cut,threshUP,threshDOWN)
#
##    # Calculation of trial results
#    perc_trials_up = []
#    perc_trials_down = []
##    mean_UP_trials = []
#    mean_DOWN_trials = []
##    cv_UP_trials = []
#    cv_DOWN_trials = []
#
#    for i in range(ntrials):# calculate results for the different trials
#        perc_up,perc_down = ratio_simtime_up_down(up_dur_trials[i],down_dur_trials[i]) # need to take trial sim length
#        mean_up,cv_up,mean_down,cv_down = mean_cv(up_dur_trials[i],down_dur_trials[i])
#
#        perc_trials_up.append(perc_up)
#        perc_trials_down.append(perc_down)
#        mean_UP_trials.append(mean_up)
#        mean_DOWN_trials.append(mean_down)
#        cv_UP_trials.append(cv_up)
#        cv_DOWN_trials.append(cv_down)
##

#    # compute the total averages across the trials taking the averages from each of the trials
##    perc_up_mean = np.mean(perc_trials_up)
#    perc_down_mean = np.mean(perc_trials_down)
#    mean_up_mean = np.mean(mean_UP_trials)
##    mean_down_mean = np.mean(mean_DOWN_trials)
#    cv_up_mean = np.mean(cv_UP_trials)
#    cv_down_mean = np.mean(cv_DOWN_trials)
#
#    # compute error bars for the calculated means
#    perc_up_stderror = np.std(perc_trials_up)/np.sqrt(ntrials)
#    perc_down_stderror = np.std(perc_trials_down)/np.sqrt(ntrials)
#    mean_up_stderror = np.std(mean_UP_trials)/np.sqrt(ntrials)
#    mean_down_stderror = np.std(mean_DOWN_trials)/np.sqrt(ntrials)
#    cv_up_stderror = np.std(cv_UP_trials)/np.sqrt(ntrials)
#    cv_down_stderror = np.std(cv_DOWN_trials)/np.sqrt(ntrials)
#
##    results = (re_mean,ri_mean,re_mean_stderror,ri_mean_stderror,re_var,ri_var,re_var_stderror,ri_var_stderror,perc_up_mean,perc_down_mean,perc_up_stderror,perc_down_stderror,mean_up_mean,mean_down_mean,mean_up_stderror,mean_down_stderror,cv_up_mean,cv_down_mean,cv_up_stderror,cv_down_stderror)
#
#    return results,up_dur_trials,down_dur_trials # added up_dur_trials and down_dur_trials to compute SCC before only had results here




#def duration_distributions(sim_time,re_trials,mov_avg_window):
#    """Calculate thresholds from firing rate and then Up and Down states to plot those distributions later"""
#    # cutting of 1s from re and ri and 200 time steps from the back of moving average
#    cut_off_start = int(1/ups_fast.dt)
#    cut_off_end = 200
#
#    # amount of trials
#    ntrials = len(re_trials)
#
    # conversion to array
#    re_trials_array = np.array(re_trials) # still need whole time series for computation of mov_avg
#    re_trials_array_cut = re_trials_array[:,cut_off_start:]
#
#    # moving average calculation based on whole time series
#    mov_avg_trials = trials.moving_avg_trials(re_trials_array,mov_avg_window)
#    mov_avg_trials_cut = mov_avg_trials[:,cut_off_start:-cut_off_end] # take away the 1s and the last 200 seconds of moving average
#    sim_time_cut = sim_time[cut_off_start:-cut_off_end]
#    # create less data for threshold determination
#    mov_avg_trials_flat = mov_avg_trials_cut.flatten()
##    no_for_less_data = np.max((1,np.math.ceil(len(mov_avg_trials_flat)/10000000)))# want to take roughly 10mio datapoints, if we have too few data points (less than 10 mio) take every datapoint.
#    mov_avg_less_data = np.msort(mov_avg_trials_flat)[::no_for_less_data] # every "no_for_less_data" we take a data point from moving avg
#
#    # Calculation of Thresholds
#    rate_up,rate_down,rate_middle,threshUP,threshDOWN = find_thresholds(mov_avg_less_data)

#    # Calculation of the UP and DOWN durations, do not need to calculate moving average as it is already done before
#    up_dur_trials,down_dur_trials,up_counts_trials,down_counts_trials,start_down_trials,up_dur_total,down_dur_total = trials.calculate_durations_from_several_trials_2thresholds_no_mov_avg(sim_time_cut,mov_avg_trials_cut,threshUP,threshDOWN)

#    return up_dur_total,down_dur_total

def periodogram(data,dt,df):
    """
    data: long 1D array of stationary process, longer than NFFT=1./(dt*df)
    Optimal is data length NFFT*Ntrials, but doesn't have to be multiple of NFFT

    NFFT is the blocklength
    L is the length of the input data
    dt is the timestep we sample the data at
    1/df is the segment length. The bigger df the smaller the segment lengths we are looking at
    segment length should contain several Ups and Downs
    """
    # calculation of blocklength (number of samples)
    NFFT=int(1./(dt*df)+0.5)
    df=1./(NFFT*dt)
    L=len(data)
    # length of the data divided by the blocklength NFFT gives us the number of trials
    # need to round it to the bottom such that the last bit of data is not used when the numbers are not divisible, otherwise I obtain an error in the reshape later
    Ntrials=math.floor(L/NFFT)
    # take the part of the data up to Ntrials*NFFT and then create a matrix with row for each trial
    x=data[:int(Ntrials*NFFT)].reshape((-1,NFFT))
    ntrials=x.shape[0]
    xF=fft(x) # Fourier Trafo
    # .conjugate() conjugates every entry in the vector/matrix. Then we multiply the conjugated matrix with the other normal matrix
    # xF*xF.conjugate() das multipliziert einfach nur jeden einzelnen eintrag mit seinem konjugierten wert
    # danach nehmen wir nur den Realteil und schmeißen den imaginärteil weg
    # wir summieren über die zeilen. also jedes mal nehmen wir den wert für die verschiedenen trials und summieren das auf
    # da wir schon im raum der frequenzen sind da wir die fourier transformierten nehmen mitteln wir jetzt alle trials was
    # rausgekommen ist für w = 0, w = df, w = 2*df,... das sind die frequenzen gegen die wir plotten
    S= np.sum(np.real(xF*xF.conjugate()),axis=0)*dt/(NFFT-1)/ntrials
    # if NFFT/2 is not an integer round it to the floor
    psd=S[1:math.floor(NFFT/2)]
    # die frequenzen gegen die wir plotten sind die dfs
    freq=df*np.arange(math.floor(NFFT/2)-1)+df

    return (freq,psd,Ntrials,df)
