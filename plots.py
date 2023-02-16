import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize, bisect

# check if csv, plots, and excel directories exist
if not os.path.exists("csv"):
    os.mkdir("csv")
if not os.path.exists("plots"):
    os.mkdir("plots")
if not os.path.exists("excel"):
    os.mkdir("excel")

# the model that is being fit to the data
def power_law(x, t):
    return 0.5 ** (x / t)

# calculate the s-statistic for a given value of t
def s_statistic(t, x, y, uncertainty):
    model = power_law(x, t)
    s = np.sum(((y - model) / uncertainty) ** 2)

    return s

# for determining error in half-thickness: find the t-values that correspond to the s-value that is 1 greater than the minimum s-value (for 0.68 confidence)
def find_t_values(x, y, uncertainty):
    res = minimize(s_statistic, t0, args=(x, y, uncertainty))
    min_s = res.fun
    t_min = res.x[0]
    target = min_s + 1
    t1 = bisect(lambda t: s_statistic(t, x, y, uncertainty) - target, t_min, t_min + 10)
    t2 = bisect(lambda t: s_statistic(t, x, y, uncertainty) - target, t_min - 10, t_min)

    return t1, t2

# apply the correction factor to the raw count rate
def calc_corrected_rate(counts, time):
    counts_per_second = counts / time
    unc_counts_per_second = np.sqrt(counts) / time
    corrected_rate = counts_per_second / (1 - (counts_per_second / 3500))
    correction_factor = 1 / (1 - (counts_per_second / 3500))
    unc_corrected_rate = correction_factor * unc_counts_per_second
    return corrected_rate, unc_corrected_rate

# subtract the background count rate from the corrected count rate
def calc_net_rate(corrected_rate, unc_corrected_rate, bg_counts, bg_time):
    corrected_rate_bg, unc_corrected_rate_bg = calc_corrected_rate(bg_counts, bg_time)
    net_rate = corrected_rate - corrected_rate_bg
    unc_net_rate = np.sqrt(unc_corrected_rate ** 2 + unc_corrected_rate_bg ** 2)
    return net_rate, unc_net_rate

# normalize the net count rate by the net count rate when there is no absorber
def calc_normalized_count_rate(
    net_rate, unc_net_rate, no_abs_counts, no_abs_time, bg_counts, bg_time
):
    corrected_rate_no_abs, unc_corrected_rate_no_abs = calc_corrected_rate(
        no_abs_counts, no_abs_time
    )
    net_rate_no_abs, unc_net_rate_no_abs = calc_net_rate(
        corrected_rate_no_abs, unc_corrected_rate_no_abs, bg_counts, bg_time
    )
    normalized_count_rate = net_rate / net_rate_no_abs
    ncr_uncertainty = (
        np.sqrt(
            (unc_net_rate / net_rate) ** 2
            + (unc_net_rate_no_abs / net_rate_no_abs) ** 2
        )
        * normalized_count_rate
    )
    return normalized_count_rate, ncr_uncertainty

# loop through all the csv files in the csv directory...
for file in os.listdir("csv"):
    print(f"Processing {file}...")
    df = pd.read_csv("csv/" + file)
    
    material = df["material"]
    thickness = df["thickness"]
    counts = df["counts"]
    time = df["time"]

    bg_counts = df["bg_counts"][0]
    bg_time = df["bg_time"][0]
    no_abs_counts = df["no_abs_counts"][0]
    no_abs_time = df["no_abs_time"][0]

    # performing the calculations
    corrected_rate, unc_corrected_rate = calc_corrected_rate(counts, time)
    net_rate, unc_net_rate = calc_net_rate(
        corrected_rate, unc_corrected_rate, bg_counts, bg_time
    )
    normalized_count_rate, ncr_uncertainty = calc_normalized_count_rate(
        net_rate, unc_net_rate, no_abs_counts, no_abs_time, bg_counts, bg_time
    )
    x, y = thickness, normalized_count_rate
    uncertainty = ncr_uncertainty
    x_max = np.max(x)
    source = file.split("_")[0]
    absorber = file.split("_")[1].split(".")[0]
    t0 = 0.1
    # find the optimal value of t
    result = minimize(s_statistic, t0, args=(x, y, uncertainty))
    # the above line is equivalent to the following commented out lines, but more accurate and efficient
    #### min_s = None
    #### t_opt = None
    #### for n in np.arange(0.01, 10.0, 0.01):
    ####     if min_s == None:
    ####         min_s = s_statistic(n, x, y, uncertainty)
    ####         t_opt = n
    ####     elif min_s != None and s_statistic(n, x, y, uncertainty) < min_s:
    ####         min_s = s_statistic(n, x, y, uncertainty)
    ####         t_opt = n
    #### print(f"Optimal value of t: {t_opt}")
    #### print(f"Minimum value of s achieved: {min_s}")
    t_opt = result.x[0]
    y_fit = power_law(np.linspace(0, x_max, 100), t_opt)
    s_min = result.fun
    
    # find the uncertainty in the optimal value of t
    t1, t2 = find_t_values(x, y, uncertainty)
    
    # calculate values of s for a range of t values
    s_values = [
        s_statistic(t, x, y, uncertainty) for t in np.linspace(t_opt -t_opt/10, t_opt + t_opt/10, 100)
    ]
    # logging...
    print("Optimal value of t:", t_opt)
    print(f"Optimal t-value uncertainty: ±{abs(t_opt - t1):.5f}")
    print("Minimum value of s achieved:", s_min)
    print(f"Expected value of s: {len(x)-1}")
    # plot the data with error bars and the best-fit model
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.errorbar(x, y, yerr=uncertainty, fmt="o", label="data", ecolor="black", ms=3)
    plt.plot(
        np.linspace(0, x_max, 100),
        y_fit,
        c="grey",
        label=r"fit, $y=(0.5)^{x/t}$" + f", t={t_opt:.3f} ± {abs(t_opt - t1):.5f}\nconfidence = 0.68",
    )
    plt.xlabel(
        "Layers of tissue" if absorber == "tissue" else f"Thickness of {absorber} (in)"
    )
    plt.ylabel("Normalized count rate")
    plt.title(f"{source} source with {absorber} absorber")
    plt.legend()
    plt.grid()

    # plot the s-value vs t curve
    plt.subplot(122)
    plt.plot(np.linspace(t_opt -t_opt/10, t_opt + t_opt/10, 100), s_values, c="grey", label="s-value vs t curve")
    plt.xlabel("t")
    plt.ylabel("s-value")
    plt.title("s-value vs t")
    plt.grid()
    # plot the minimum value of s achieved and the two points used to calculate the uncertainty in t
    plt.plot(t_opt, result.fun, "ro", label=f"Lowest s value: {result.fun:.3f}")
    plt.plot(t1, s_statistic(t1, x, y, uncertainty),c="black", marker="o")
    plt.plot(t2, s_statistic(t2, x, y, uncertainty), c="black", marker="o")
    # plot the expected value of s as a line
    plt.axhline(len(x) - 1, color="cornflowerblue", label=f"Expected value of s: {len(x) - 1}")
    plt.legend()
    
    # save the plot to file
    plt.savefig(f"plots/{source}_{absorber}.png")
    
    # save relevant information to an excel file
    df_data = pd.DataFrame(
        {
            "Source": [source] + [np.nan] * (len(x) - 1),
            "Material": absorber,
            "Absorber Letter": material,
            "Absorber Thickness": thickness,
            "Time": time,
            "Counts": counts,
            "NCR": normalized_count_rate,
            "NCR Uncertainty": ncr_uncertainty,
            "Minimized S-Value": [result.fun] + [np.nan] * (len(x) - 1),
            "Optimal Half-Thickness": [t_opt] + [np.nan] * (len(x) - 1),
            "Optimal Half-Thickness Uncertainty": [f"±{abs(t_opt - t1)}"]
            + [np.nan] * (len(x) - 1),
        }
    )
    df_data.to_excel(f"excel/{source}_{absorber}.xlsx", index=False)
