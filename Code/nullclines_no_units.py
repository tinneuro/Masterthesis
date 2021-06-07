import numpy as np
import matplotlib.pyplot as plt

# script for the nullcline plots
r_max = 100


def sigmoidal_transfer(h,g,theta):
    """Sigmoidal transfer function with linear part setting to 0 Hz below 2mV"""
    phi = r_max/(1 + np.exp(-g*(h-theta)))
    return phi

def inverse_sigmoidal(r,g,theta):
    """Inverse sigmoidal transfer function
    Need to be careful not possible to input >=100Hz or <=0 Hz."""
    h = -1/g*np.log((r_max - r)/r) + theta
    return h

def excitatory_nullcline_hi(h_e,a,params):
    """Calculation of the excitatory Nullcline for the sigmoidal case.
    We have conditions for different values of r to input well-defined values for inverse sigmoidal."""
    w_EE,w_EI,w_IE,w_II,g_e,g_i,theta_e,theta_i,I_E,I_I = params
    r = (w_EE*sigmoidal_transfer(h_e,g_e,theta_e) - h_e + I_E - a)/w_EI
    if (r < 0) or (r > 100):
        # for these cases the inverse sigmoidal is not defined
        h_i = np.nan
    else:
        h_i = inverse_sigmoidal(r,g_i,theta_i)
    return h_i

def inhibitory_nullcline(h_i,params):
    """Calculation of the inhibitory Nullcline for the sigmoidal case.
    Input and return opposite as normal, but we plot it the other way round.
    We have conditions for different values of r to input well-defined values for inverse sigmoidal"""
    w_EE,w_EI,w_IE,w_II,g_e,g_i,theta_e,theta_i,I_E,I_I = params
    r = (h_i + w_II*sigmoidal_transfer(h_i,g_i,theta_i) - I_I)/w_IE
    if (r < 0) or (r > 100):
        # for these cases the inverse sigmoidal is not defined
        h_e = np.nan
    else:
        h_e = inverse_sigmoidal(r,g_e,theta_e)
    return h_e

def exc_nullcline_adiabatic_elim(h,params):
    """Excitatory nullcline after adiabatic elimination."""
    w_EE,w_EI,w_IE,w_II,g_e,g_i,theta_e,theta_i,I_E,I_I = params
    F_E = sigmoidal_transfer(h,g_e,theta_e)
    a = -h + w_EE*F_E - w_EI*sigmoidal_transfer(w_IE*F_E + I_I,g_i,theta_i) + I_E
    return a

def a_nullcline(h,params,beta):
    """Nullcline for the Adaptation does not matter whether adiabatic elimination or not.
    This nullcline stays the same."""
    w_EE,w_EI,w_IE,w_II,g_e,g_i,theta_e,theta_i,I_E,I_I = params
    a = beta*sigmoidal_transfer(h,g_e,theta_e)
    return a

def interactive_he_hi_nullclines(a = 1.2,w_EE = 2.4,w_EI = 1.5,w_IE = 2.33,w_II = 0,g_e =120,g_i= 350,theta_e = 35,theta_i = 35,I_E = 0,I_I = 0):
    # input in mV transform it here to V
    a = a/1000
    w_EE = w_EE/1000
    w_EI = w_EI/1000
    w_IE = w_IE/1000
    w_II = w_II/1000
    theta_e = theta_e/1000
    theta_i = theta_i/1000
    I_E = I_E/1000
    I_I = I_I/1000

    params_interact = [w_EE,w_EI,w_IE,w_II,g_e,g_i,theta_e,theta_i,I_E,I_I]
    h_input_inh = np.linspace(-0.01,0.04,200)
    h_input_exc = np.linspace(-0.04,0.04,6000)

    F_E = sigmoidal_transfer(h_input_exc,g_e,theta_e)
    F_I = sigmoidal_transfer(h_input_inh,g_i,theta_i)

    # calculate the nullclines
    exc_nullcline = []
    inh_nullcline = []

    for h in h_input_exc:
        exc_nullcline.append(excitatory_nullcline_hi(h,a,params_interact))
    for h in h_input_inh:
        inh_nullcline.append(inhibitory_nullcline(h,params_interact))


    # Plotting
    plt.figure(figsize=(20,10))
    plt.ylim((-0.003,0.04))
    plt.xlim((-0.03,0.04))
    plt.plot(h_input_exc,exc_nullcline,label = "Excitatory Nullcline",color = "#E24A33")
    plt.plot(inh_nullcline,h_input_inh,label = "Inhibitory Nullcline",color = "#348ABD")
    plt.xlabel("he in Volt")
    plt.ylabel("hi in Volt")
    plt.title("Exc and Inh Nullclines")
    plt.legend()


def interactive_nullclines_adiab_elim(beta = 0.4,w_EE = 2.4,w_EI = 1.5,w_IE = 2.3,w_II = 0,g_e = 0.12,g_i= 0.35,theta_e = 35,theta_i = 35,I_E = 0,I_I = 0):
    """Interactive Nullclines 3D model adiabatic elimination """
    params_interact = [w_EE,w_EI,w_IE,w_II,g_e,g_i,theta_e,theta_i,I_E,I_I]
    h_input_exc = np.linspace(-40,40,6000)
    F_E = sigmoidal_transfer(h_input_exc,g_e,theta_e)

    he_nullcline_adiab_elim = exc_nullcline_adiabatic_elim(h_input_exc,params_interact)
    a_nullcline_he = a_nullcline(h_input_exc,params_interact,beta)

    plt.plot(he_nullcline_adiab_elim,F_E,label = "Excitatory Nullcline")
    plt.plot(a_nullcline_he,F_E,label = "Adaptation Nullcline")
    plt.xlabel("Adaptation mV")
    plt.ylabel("$r_E$ Hz")
    plt.xlim(-20,20)
    plt.ylim(0,20)
    plt.legend()
