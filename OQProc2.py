"""
Utility functions used to:
    post-process OpenQuake PSHA results
    read record files
    create design spectra from building codes
    to retrieve ESM database token to download records
    to check available gmpes in OpenQuake and their attributes
"""


#############################################################################################
################### Methods to post-process OpenQuake PSHA results ##########################
#############################################################################################

def hazard(poes, path_hazard_results, output_dir='Outputs', rlz='hazard_curve-mean', i_save=0, i_show=0):
    import pandas as pd
    import numpy as np
    from scipy import interpolate
    import os
    import matplotlib.pyplot as plt
    """
    Details
    -------
    This script will plot the hazard curve

    Parameters
    ----------
    poes : list
        Probabilities of exceedance in tw years for which im levels will be obtained.
    path_hazard_results: str
        Path to the hazard results
    output_dir: str, optional
        Save outputs
    rlz : str, optional
        realization name to plot.

    Returns
    -------
    None.

    """

    # Initialise some lists
    lat = []
    lon = []
    im = []
    s = []
    poe = []
    apoe = []
    id_no = []
    imls = []

    # Read through each file in the outputs folder
    count = 0
    for file in os.listdir(path_hazard_results):
        if file.startswith(rlz):
            # Strip the IM out of the file name
            im_type = (file.rsplit('-')[2]).rsplit('_')[0]
            # Load the results in as a dataframe
            df = pd.read_csv(''.join([path_hazard_results, '/', file]), skiprows=1)
            # Get the column headers (but they have a 'poe-' string in them to strip out)
            iml = list(df.columns.values)[3:]  # List of headers
            iml = [float(i[4:]) for i in iml]  # Strip out the actual IM values
            f = open(''.join([path_hazard_results, '/', file]), "r")
            temp1 = f.readline().split(',')
            temp2 = list(filter(None, temp1))
            inv_t = float(temp2[5].replace(" investigation_time=", ""))
            f.close()

            # Append each site's info to the output array
            lat.append([df.lat[0]][0])
            lon.append([df.lon[0]][0])
            im.append(im_type)
            s.append(iml)
            # Get the array of poe in inv_t
            poe.append(df.iloc[0, 3:].values)

            # For each array of poe, convert it to annual poe
            temp = []
            for i in np.arange(len(poe[-1])):
                temp.append(-np.log(1 - poe[-1][i]) / inv_t)
            apoe.append(temp)
    # Get intensity measure levels corresponding to poes
    for i in range(len(s)):
        plt.figure(figsize=(6.4, 5.2))
        plt.loglog(s[i], poe[i], label=im[i], color='salmon', lw=1, alpha=0.95)
        Ninterp = 1e5
        iml_range = np.arange(min(s[i]), max(s[i]), (max(s[i]) - min(s[i])) / Ninterp)
        poe_fit = interpolate.interp1d(s[i], np.asarray(poe[i]), kind='quadratic')(iml_range)
        idxs = []
        for ij in range(len(poes)):
            temp = abs(poe_fit - poes[ij]).tolist()
            idxs.append(temp.index(min(temp)))
            # These are actual points where the analysis are carried out and losses are calculated for
        iml = iml_range[idxs]
        imls.append(iml)
        plt.rcParams["font.family"] = "Times New Roman"
        csfont = {'fontname': 'Times New Roman'}
        plt.tick_params(axis="x", labelsize=11)
        plt.tick_params(axis="y", labelsize=11)
        plt.ylim([0.001, 2])
        plt.xlim([0.01, 3])
        plt.xlabel(str(im[i]) + '  [g]', fontsize=14, **csfont)
        plt.ylabel('Probability of exceedance in 50 years', fontsize=14, **csfont)
        plt.gca().spines["top"].set_alpha(.3)
        plt.gca().spines["bottom"].set_alpha(.3)
        plt.gca().spines["right"].set_alpha(.3)
        plt.gca().spines["left"].set_alpha(.3)
        plt.grid(True, which="both", color='gainsboro', alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(output_dir, 'Hazard_Curve_' + im[i] + '.png')
        if i_save == 1:
            plt.savefig(fname, format='png', dpi=900)
        if i_show == 1:
            plt.show()
    return imls


def hazard_curve(poes, path_hazard_results, output_dir='Post_Outputs', filename='hazard_curve-mean'):
    """
    Details
    -------
    This script will save hazard curves and  iml's corresponding to the desired poes
    as .txt files, and the plot the hazard curves in the same figure.
    Parameters
    ----------
    poes : list
        Probabilities of exceedance in tw years for which im levels will be obtained.
    path_hazard_results: str
        Path to the hazard results.
    output_dir: str, optional
        Save outputs to a pickle file.
    filename : str, optional
        filename to process.
    Returns
    -------
    None.
    """

    import pandas as pd
    import numpy as np
    from scipy import interpolate
    import os
    import matplotlib.pyplot as plt

    def get_iml(poes, apoe_data, iml_data, inv_t):
        """
        Details
        -------
        This script will take results of OpenQuake PSHA analysis, and return
        the intensity measure levels for desired probability of exceedance values.
        Parameters
        ----------
        poes: list
            desired probability of exceedance values to calculate their
            corresponding intensity measure levels.
        apoe_data: list
            annual probability of exceedance values.
        iml_data: list
            intensity measure levels.
        inv_t: int
            investigation time.
        Returns
        -------
        iml: list
            intensity measure levels corresponding to poes.
        """

        infs = np.isinf(apoe_data)
        apoe_data = apoe_data[~infs]
        iml_data = iml_data[~infs]
        nans = np.isnan(apoe_data)
        apoe_data = apoe_data[~nans]
        iml_data = iml_data[~nans]

        Ninterp = 1e5
        iml_range = np.arange(min(iml_data), max(iml_data), (max(iml_data) - min(iml_data)) / Ninterp)
        apoe_fit = interpolate.interp1d(iml_data, apoe_data, kind='quadratic')(iml_range)
        poe = 1 - (1 - apoe_fit) ** inv_t

        idxs = []
        for i in range(len(poes)):
            temp = abs(poe - poes[i]).tolist()
            idxs.append(temp.index(min(temp)))
            # These are actual points where the analysis are carried out and losses are calculated for
        iml = iml_range[idxs]

        return iml

    # Initialise some lists
    lat = []
    lon = []
    im = []
    s = []
    poe = []
    apoe = []
    id_no = []
    imls = []

    # Read through each file in the outputs folder
    for file in os.listdir(path_hazard_results):
        if file.startswith(filename):

            # print(file)
            # Strip the IM out of the file name
            im_type = (file.rsplit('-')[2]).rsplit('_')[0]

            # Get the id number of the file
            idn = (file.rsplit('_')[2]).rsplit('.')[0]

            # Load the results in as a dataframe
            df = pd.read_csv(''.join([path_hazard_results, '/', file]), skiprows=1)

            # Get the column headers (but they have a 'poe-' string in them to strip out)
            iml = list(df.columns.values)[3:]  # List of headers
            iml = [float(i[4:]) for i in iml]  # Strip out the actual IM values
            f = open(''.join([path_hazard_results, '/', file]), "r")
            temp1 = f.readline().split(',')
            temp2 = list(filter(None, temp1))
            inv_t = float(temp2[5].replace(" investigation_time=", ""))
            f.close()

            # For each of the sites investigated
            for site in np.arange(len(df)):

                # Append each site's info to the output array
                lat.append([df.lat[site]][0])
                lon.append([df.lon[site]][0])
                im.append(im_type)
                s.append(iml)
                id_no.append(idn)

                # Get the array of poe in inv_t
                poe.append(df.iloc[site, 3:].values)

                # For each array of poe, convert it to annual poe
                temp = []
                for i in np.arange(len(poe[-1])):
                    temp.append(-np.log(1 - poe[-1][i]) / inv_t)
                apoe.append(temp)

    # Get intensity measure levels corresponding to poes
    fig = plt.figure()
    for i in range(len(s)):
        plt.loglog(s[i], apoe[i], label=im[i])
        iml = get_iml(np.asarray(poes), np.asarray(apoe[i]), np.asarray(s[i]), inv_t)
        imls.append(iml)
        fname = os.path.join(output_dir, 'imls_' + im[i] + '.out')
        f = open(fname, 'w+')
        for j in iml:
            f.write("%.3f\n" % j)
        f.close()

    fname = os.path.join(output_dir, 'poes.out')
    f = open(fname, 'w+')
    for j in poes:
        f.write("%.4f\n" % j)
    f.close()

    plt.xlabel('IM [g]')
    plt.ylabel('Annual Probability of Exceedance')
    plt.legend()
    plt.grid(True)
    plt.title(f"Mean Hazard Curves for Lat:{lat[0]} Lon:{lon[0]}")
    plt.tight_layout()
    fname = os.path.join(output_dir, 'Hazard_Curves.png')
    plt.savefig(fname, format='png', dpi=220)
    plt.show()
    plt.close(fig)

    for i in range(len(apoe)):
        poe = 1 - (1 - np.asarray(apoe[i])) ** inv_t
        poe.shape = (len(poe), 1)
        imls = np.asarray(s[i])
        imls.shape = (len(imls), 1)
        haz_cur = np.concatenate([imls, poe], axis=1)
        fname = os.path.join(output_dir, 'HazardCurve_' + im[i] + '.out')
        np.savetxt(fname, haz_cur)


def disagg_MR(Mbin, dbin, path_disagg_results, output_dir='Post_Outputs', n_rows=1, filename='Mag_Dist'):
    """
    Details
    -------
    This script will save disaggregation plots including M and R.
    Parameters
    ----------
    Mbin : int, float
        magnitude bin used in disaggregation.
    dbin : int, float
        distance bin used in disaggregation.
    path_disagg_results: str
        Path to the disaggregation results.
    output_dir: str, optional
        Save outputs to a pickle file.
    n_rows : int, optional
        total number of rows for subplots.
    filename : str, optional
        filename to process.
    Returns
    -------
    None.
    """

    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm  # import colormap

    # lets add the plotting options to make everything clearer
    cmap = cm.get_cmap('jet')  # Get desired colormap

    for file in os.listdir(path_disagg_results):
        if file.startswith(filename) and 'Mag_Dist_Eps' not in file:
            # Load the dataframe
            df = pd.read_csv(''.join([path_disagg_results, '/', file]), skiprows=1)
            poes = np.unique(df['poe']).tolist()
            poes.sort(reverse=True)
            # Get some salient values
            f = open(''.join([path_disagg_results, '/', file]), "r")
            ff = f.readline().split(',')
            lon = float(ff[-2].replace(" lon=", ""))
            lat = float(ff[-1].replace(" lat=", "").replace("\"\n", ""))
            ims = np.unique(df['imt'])
            inv_t = float(ff[7].replace(" investigation_time=", ""))
            for imt in ims:
                M, R = [], []
                apoe_norm = []
                Tr = []
                modeLst, meanLst = [], []
                for poe in poes:
                    Tr.append(round(-inv_t / np.log(1 - poe)))
                    data = {}
                    data['mag'] = df['mag'][(df['poe'] == poe) & (df['imt'] == imt)]
                    data['dist'] = df['dist'][(df['poe'] == poe) & (df['imt'] == imt)]
                    data['apoe'] = -np.log(1 - df.iloc[:, 4][(df['poe'] == poe) & (df['imt'] == imt)]) / inv_t
                    apoe_norm.append(data['apoe'] / data['apoe'].sum())
                    data['apoe_norm'] = apoe_norm[-1]
                    data = pd.DataFrame(data)
                    # Compute the modal value (highest apoe)
                    mode = data.sort_values(by='apoe_norm', ascending=False)[0:1]
                    modeLst.append([mode['mag'].values[0], mode['dist'].values[0]])
                    # Compute the mean value
                    meanLst.append([np.sum(data['mag'] * data['apoe_norm']), np.sum(data['dist'] * data['apoe_norm'])])

                    # Report the individual mangnitude and distance bins
                    M.append(data['mag'])
                    R.append(data['dist'])

                n_Tr = len(Tr)
                mags = []
                dists = []

                n_cols = int(np.floor(n_Tr / n_rows))
                if np.mod(n_Tr, n_rows):
                    n_cols += 1

                fig = plt.figure(figsize=(19.2, 10.8))
                for i in range(n_Tr):
                    ax1 = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')

                    X = R[i]
                    Y = M[i]
                    Z = np.zeros(len(X))

                    dx = np.ones(len(X)) * dbin / 2
                    dy = np.ones(len(X)) * Mbin / 2
                    dz = apoe_norm[i] * 100

                    # here we may make the colormap based on epsilon instead of hazard contribution
                    max_height = np.max(dz)  # get range of color bars so we can normalize
                    min_height = np.min(dz)
                    # scale each z to [0,1], and get their rgb values
                    rgba = [cmap((k - min_height) / max_height) for k in dz]
                    ax1.bar3d(X, Y, Z, dx, dy, dz, color=rgba, zsort='average', alpha=0.7, shade=True)

                    ax1.set_xlabel('R [km]')
                    ax1.set_ylabel('$M_{w}$')
                    if np.mod(i + 1, n_cols) == 1:
                        ax1.set_zlabel('Hazard Contribution [%]')
                        ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
                        ax1.set_zlabel('Hazard Contribution [%]', rotation=90)
                    ax1.zaxis._axinfo['juggled'] = (1, 2, 0)

                    plt.title('$T_{R}$=%s years\n$M_{mod}$=%s, $R_{mod}$=%s km\n$M_{mean}$=%s, $R_{mean}$=%s km'
                              % ("{:.0f}".format(Tr[i]), "{:.2f}".format(modeLst[i][0]), "{:.0f}".format(modeLst[i][1]),
                                 "{:.2f}".format(meanLst[i][0]), "{:.0f}".format(meanLst[i][1])),
                              fontsize=11, loc='right', verticalalignment='top', y=0.95)

                    mags.append(meanLst[i][0])
                    dists.append(meanLst[i][1])

                plt.subplots_adjust(hspace=0.05, wspace=0.05)  # adjust the subplot to the right for the legend
                fig.suptitle(f"Disaggregation of Seismic Hazard\nIntensity Measure: {imt}\nLatitude: "
                             f"{lat:.4f}, Longitude: {lon:.4f}", fontsize=14, weight='bold', ha='left', x=0.0, y=1.0)

                plt.tight_layout(rect=[0, 0.0, 1, 0.94])
                fname = os.path.join(output_dir, 'Disaggregation_MR_' + imt + '.png')
                plt.savefig(fname, format='png', dpi=220)

                fname = os.path.join(output_dir, 'mean_mags_' + imt + '.out')
                np.savetxt(fname, np.asarray(mags), fmt='%.2f')
                fname = os.path.join(output_dir, 'mean_dists_' + imt + '.out')
                np.savetxt(fname, np.asarray(dists), fmt='%.1f')
                plt.show()
                plt.close(fig)


def disagg_MReps(Mbin, dbin,imtiv, path_disagg_results, output_dir='Post_Outputs', n_rows=1, filename='Mag_Dist_Eps'):
    """
    Details
    -------
    This script will save disaggregation plots
    including M and R.
    Parameters
    ----------
    Mbin : int, float
        magnitude bin used in disaggregation.
    dbin : int, float
        distance bin used in disaggregation.
    path_disagg_results: str
        Path to the hazard results
    output_dir: str, optional
        Save outputs to a pickle file
    n_rows : int, optional
        total number of rows for subplots.
    filename : str, optional
        filename to process.
    Returns
    -------
    None.
    """

    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm  # import colormap
    from matplotlib.patches import Patch

    # lets add the plotting options to make everything clearer
    cmap = cm.get_cmap('jet')  # Get desired colormap

    mags = []
    dists = []
    
    # check the imt we are selecting for
    if imtiv==0:
        imTi="PGA"
    else:
        imTi="SA("+str(imtiv)+")"

    for file in os.listdir(path_disagg_results):
        if file.startswith(filename):
            # Load the dataframe
            df = pd.read_csv(''.join([path_disagg_results, '/', file]), skiprows=1)
            poes = np.unique(df['poe']).tolist()
            poes.sort(reverse=True)
            # Get some salient values
            f = open(''.join([path_disagg_results, '/', file]), "r")
            ff = f.readline().split(',')
            lon = float(ff[-2].replace(" lon=", ""))
            lat = float(ff[-1].replace(" lat=", "").replace("\"\n", ""))
            ims = np.unique(df['imt'])
            inv_t = float(ff[8].replace(" investigation_time=", ""))
            for imt in ims:
                modeLst, meanLst = [], []
                Tr = []
                apoe_norm = []
                M, R, eps = [], [], []
                for poe in poes:
                    Tr.append(round(-inv_t / np.log(1 - poe)))
                    data = {}
                    data['mag'] = df['mag'][(df['poe'] == poe) & (df['imt'] == imt)]
                    data['eps'] = df['eps'][(df['poe'] == poe) & (df['imt'] == imt)]
                    data['dist'] = df['dist'][(df['poe'] == poe) & (df['imt'] == imt)]
                    data['apoe'] = -np.log(1 - df.iloc[:, 5][(df['poe'] == poe) & (df['imt'] == imt)]) / inv_t
                    apoe_norm.append(np.array(data['apoe'] / data['apoe'].sum()))
                    data['apoe_norm'] = apoe_norm[-1]
                    data = pd.DataFrame(data)
                    # Compute the modal value (highest apoe)
                    mode = data.sort_values(by='apoe_norm', ascending=False)[0:1]
                    modeLst.append([mode['mag'].values[0], mode['dist'].values[0], mode['eps'].values[0]])
                    # Compute the mean value
                    meanLst.append([np.sum(data['mag'] * data['apoe_norm']), np.sum(data['dist'] * data['apoe_norm']),
                                    np.sum(data['eps'] * data['apoe_norm'])])

                    # Report the individual mangnitude and distance bins
                    M.append(np.array(data['mag']))
                    R.append(np.array(data['dist']))
                    eps.append(np.array(data['eps']))
                    if imTi==imt and poe==max(poes):
                        modeLstout=modeLst
                        meanLstout=meanLst
                        

                n_Tr = len(Tr)
                n_eps = len(np.unique(np.asarray(eps)))
                min_eps = np.min(np.unique(np.asarray(eps)))  # get range of colorbars so we can normalize
                max_eps = np.max(np.unique(np.asarray(eps)))

                n_cols = int(np.floor(n_Tr / n_rows))
                if np.mod(n_Tr, n_rows):
                    n_cols += 1

                fig = plt.figure(figsize=(19.2, 10.8))
                for i in range(n_Tr):
                    ax1 = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')

                    # scale each eps to [0,1], and get their rgb values
                    rgba = [cmap((k - min_eps) / max_eps / 2) for k in (np.unique(np.asarray(eps)))]
                    num_triads_M_R_eps = len(R[i])
                    Z = np.zeros(int(num_triads_M_R_eps / n_eps))

                    for l in range(n_eps):
                        X = np.array(R[i][np.arange(l, num_triads_M_R_eps, n_eps)])
                        Y = np.array(M[i][np.arange(l, num_triads_M_R_eps, n_eps)])

                        dx = np.ones(int(num_triads_M_R_eps / n_eps)) * dbin / 2
                        dy = np.ones(int(num_triads_M_R_eps / n_eps)) * Mbin / 2
                        dz = np.array(apoe_norm[i][np.arange(l, num_triads_M_R_eps, n_eps)]) * 100

                        ax1.bar3d(X, Y, Z, dx, dy, dz, color=rgba[l], zsort='average', alpha=0.7, shade=True)
                        Z += dz  # add the height of each bar to know where to start the next

                    ax1.set_xlabel('R [km]')
                    ax1.set_ylabel('$M_{w}$')
                    if np.mod(i + 1, n_cols) == 1:
                        ax1.set_zlabel('Hazard Contribution [%]')
                        ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
                        ax1.set_zlabel('Hazard Contribution [%]', rotation=90)
                    ax1.zaxis._axinfo['juggled'] = (1, 2, 0)

                    plt.title("$T_{R}$=%s years\n$M_{mod}$=%s, $R_{mod}$=%s km, $\epsilon_{mod}$=%s"
                              "\n$M_{mean}$=%s, $R_{mean}$=%s km, $\epsilon_{mean}$=%s"
                              % ("{:.0f}".format(Tr[i]), "{:.2f}".format(modeLst[i][0]), "{:.0f}".format(modeLst[i][1]),
                                 "{:.1f}".format(modeLst[i][2]),
                                 "{:.2f}".format(meanLst[i][0]), "{:.0f}".format(meanLst[i][1]),
                                 "{:.1f}".format(meanLst[i][2])),
                              fontsize=11, loc='right', va='top', y=0.95)

                    mags.append(meanLst[i][0])
                    dists.append(meanLst[i][1])

                legend_elements = []
                for j in range(n_eps):
                    legend_elements.append(Patch(facecolor=rgba[n_eps - j - 1],
                                                 label=f"\u03B5 = {np.unique(np.asarray(eps))[n_eps - j - 1]:.2f}"))

                fig.legend(handles=legend_elements, loc="lower center", borderaxespad=0., ncol=n_eps)
                plt.subplots_adjust(hspace=0.05, wspace=0.05)  # adjust the subplot to the right for the legend
                fig.suptitle(f"Disaggregation of Seismic Hazard\nIntensity Measure: {imt}\nLatitude: "
                             f"{lat:.4f}, Longitude: {lon:.4f}", fontsize=14, weight='bold', ha='left', x=0.0, y=1.0)
                plt.tight_layout(rect=[0, 0.03, 1, 0.94])
                fname = os.path.join(output_dir, 'Disaggregation_MReps_' + imt + '.png')
                plt.savefig(fname, format='png', dpi=220)

                fname = os.path.join(output_dir, 'mean_mags_' + imt + '.out')
                np.savetxt(fname, np.asarray(mags), fmt='%.2f')
                fname = os.path.join(output_dir, 'mean_dists_' + imt + '.out')
                np.savetxt(fname, np.asarray(dists), fmt='%.1f')
                plt.show()
                plt.close(fig)
                
    return meanLstout, modeLstout

