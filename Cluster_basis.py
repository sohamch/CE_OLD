from onsager import crystal, cluster, supercell
import numpy as np
import collections

def createSpins(mobile_count, spec_count=0):

    """
    Intended use - mobile spins symmetric about zero - make spec spins positive after that
    """
    m = (len(mobile_occ) + spec_count) // 2
    # The number of species in 2m+1 or 2m.

    spins_mobile = np.array(list(range(-m, m + 1)))
    spins_spec = None
    if spec_count > 0:
        spins_spec = np.array(list(range(m+1, m+spec_count+1)))

    if m % 2 == 1:
        spins.remove(0)  # zero not required here

    return spins_mobile, spins_spec


def createClusterBasis(sup, clusexp, mobileOccList, spins_mobile, SpecOccList=None, spins_spec=None):
    """
    Function to generate mobile site-based reduced cluster basis. Whole basis returned if no spectator sites
    Output - mobileClusterbasis - len(Rvectlist)xlen(mobilebasis) array each element contains the norm of the basis
    functions. Since the spectator sites do not move, this needs to be done only once.
    """
    if spins_spec is None and SpecOccList is not None:
        raise ValueError("The spectator sites have non-zero occupancies but no spins given.")

    # Generate the mobile basis
    m = len(spins_mobile)
    if spins_spec is not None:
        mobClusterset = collections.defaultdict(list)
        mobClusterList = []
        clustindDict = {}
        for clust in [cl for clist in clusexp for cl in clist]:
            # get the mobile sites out of this cluster and form a new cluster
            mobClust = cluster.Cluster([site for site in clust.sites if site.ci in sup.mobileindices])
            mobClusterset[mobClust].append(clust) # store the clusters where clust is present
            clustindDict[mobClust] = len(mobClusterList)
            mobClusterList.append(mobClust)  # The order of the mobile clusters in this list will be the order in
            # which they appear in the columns of the final output array
        # Now, extend the mobile clusters across the whole lattice and store the normalizations
        mobileClusterbasis = np.zeros((len(sup.Rvectlist), len(mobClusterset.items())))
        for Rind, R in enumerate(sup.Rvectlist):
            for clustind, clust in enumerate(mobClusterList):
                lambda_clust = 0
                for FullClust in mobClusterset[clust]:
                    # Extracting the spins for the spectator sites in each cluster where clust is present.
                    spinprod = np.prod(np.array([sum([spin*occlist[idx]
                                                      for spin, occlist in zip(spins_spec, SpecOccList)])
                                                 for idx in [sup.index(R, site.ci) for site in FullClust.sites
                                                             if site in sup.indexspectator]]))
                    lambda_clust += spinprod
                lambda_clust *= (m*(m+1)*(2*m+1)//3)**len(clust.sites)
                mobileClusterbasis[Rind, clustind] = lambda_clust

        return mobileClusterbasis, mobClusterList, clustindDict
    else:
        clustList = [cl for clist in clusexp for cl in clist]
        clustindDict = {}
        mobileClusterbasis = np.zeros((len(sup.Rvectlist), len(clustList)))
        for Rind, R in enumerate(sup.Rvectlist):
            for clustind, clust in enumerate(clustList):
                lambda_clust = (m * (m + 1) * (2 * m + 1) // 3) ** len(clust.sites)
                clustindDict[clust] = clustind
                mobileClusterbasis[Rind, clustind] = lambda_clust

        return mobileClusterbasis, clustList, clustindDict


def FormLBAM(sup, spins_mobile, mobileOccList, mobileClusterbasis, clustIndDict, transitions):
    """
    Function to get the LBAM expansion terms for the current state given by mobileOccList
    :param transitions; tuple containing (ijlist, ratelist, dxlist)
    :param clustIndDict: {cluster: column index of cluster in mobileClusterBasis}
    :param mobileOccList - List of occupancy vectors for the various species on the mobile sublattice in the present
    state.
    :param mobileClusterbasis: len(Rvectlist)xNBasisClusters array
    :param sup: supercell object.
    :param spins_mobile: spins on the occupied mobile sites, given by mobileOccList
    """

    ijlist, ratelist, dxlist = transitions
    # First, we need to enumerate all the ways we can leave the present state
    init2finSiteDict = collections.defaultdict(list)
    for jmp in ijlist:
        init2finSiteDict[jmp[0]].append(jmp1)
    Ntrans = len(clustIndDict.items())
    NrepClusts = len(clustIndDict.items())
    Wbar = np.zeros((Ntrans*NrepClusts+1, Ntrans*NrepClusts+1))  # +1 for the constant term
    Bbar = np.zeros(Ntrans*NrepClusts+1)

    for (initState, listFinState) in init2finSiteDict.items():
        # First, we must get the clusters in which initState belongs.
        # ciR in sup - mapping from a site index to an actual state - as tuple ((c,i), R).
        site_i = sup.ciR(initState)
        sites_j = [sup.ciR(finstate) for finstate in listFinState]
        # Get the clusters the initial state belongs to, and the translations that take the cluster there.
        # We have to find the representative clusters the initial state is a part of.
        # We have to translate the sites in that cluster so that the representative site matches up with the initial
        # site.
        # The function sup.index(R, ci) takes care of periodic boundary conditions too.
        # For a site R, ci - get the index using sup.ciR(sup.index(R, ci)) to get the image in the supercell.
        initClustList = [(cl, clind) for cl, clind in clustIndDict.items() if site_i[0] in set([site.ci
                                                                                                for site in cl.sites])]
        initClustIndlist = []
        for clust, clustind in initClustList:
            for site in clust.sites:
                if site.ci == site_i[0]:
                    # Get the translated sites
                    Rtrans = site_i[1] - site.R
                    # Get the index of this translation in Rvectlist
                    RtransInd = sup.Rvectind[(Rtrans[0], Rtrans[1], Rtrans[2])]
                    initClustIndlist.append(RtransInd*NrepClusts + clustind)
        for sites_j in sites_j:
            finClustList = [(cl, clind) for cl, clind in clustIndDict.items() if site_j[0] in set([site.ci
                                                                                                   for site in
                                                                                                   cl.sites])]
            finClustIndlist = []
            for clust, clustind in finClustList:
                for site in clust.sites:
                    if site.ci == site_j[0]:
                        # Get the translated sites
                        Rtrans = site_j[1] - site.R
                        # Get the index of this translation in Rvectlist
                        RtransInd = sup.Rvectind[(Rtrans[0], Rtrans[1], Rtrans[2])]
                        finClustIndlist.append(RtransInd * NrepClusts + clustind)






def ClusterProducts(sup, clusexp, mobilelist, spins):
    """

    :param sup: supercell object, which is also the current state of the object
    :param clusexp: list of clusters around supercell
    :param mobilelist: list of occupancies on mobile sites for each species that can occupy those sites
    "param spins : the spins of the chemical species - assume the spins of mobile species are stored first.
    :return: one-d array : corresponding to the values of the cluster basis functions for the different clusters.
    This needs to be used more than once, since the bias needs to be updated at every state.
    """

    if not np.allclose(sum([arr for arr in mobilelist]), 1):
        raise ValueError("The occupancies of sites do not add up to one")

    Nclusters = len(clist)
    PhiMobileArray = np.zeros((len(sup.Rvectlist), Nclusters))

    # Go through the position vectors
    for Rind, R in enumerate(sup.Rvectlist):
        # Go through the clusters in the representatives
        for clInd, cl in enumerate([cl for clist in clusexp for cl in clist]):
            # add the translation to all the sites in the cluster and get the indices of those sites
            # Verify this
            sitelist = [sup.index(R, site)[0] for site in cl.sites]
            # Next, get the spin products of the sites in sitelist
            Phi_mob = np.prod(np.array([sum([spin * mob[ind] for spin, spec in zip(spins, mobilelist)])
                                        for ind in sitelist]))
            # Now store the spin products in the array

            PhiMobileArray[Rind, clInd] = Phi_mob

    return PhiMobileArray

# Todo - Can we make the above computations more efficient. Imganine a case with a large nuumber of sites.
# As we go from one state to another, only a few clusters change. So the Phi array for the next state
# Can be evaluated from the Phi array of the current state easily. Check how.