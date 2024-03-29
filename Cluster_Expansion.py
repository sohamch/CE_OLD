from onsager import cluster
import numpy as np
import collections
import itertools
import Transitions
import time


class ClusterSpecies():

    def __init__(self, specList, siteList):
        if len(specList)!= len(siteList):
            raise ValueError("Species and site lists must have same length")
        if not all(isinstance(site, cluster.ClusterSite) for site in siteList):
            raise TypeError("The sites must be entered as clusterSite object instances")
        # Form (site, species) set
        # Calculate the translation to bring center of the sites to the origin unit cell
        self.specList = specList
        self.siteList = siteList
        Rtrans = sum([site.R for site in siteList])//len(siteList)
        self.transPairs = [(site-Rtrans, spec) for site, spec in zip(siteList, specList)]
        # self.transPairs = sorted(self.transPairs, key=lambda x: x[1])
        self.SiteSpecs = set(self.transPairs)
        # self.__hashcache__ = int(hash(tuple([hash((site, spec)) for site, spec in self.transPairs])))
        # self.__hashcache__ = int(np.prod(np.array([hash((site, spec)) for site, spec in self.transPairs]))) +\
        #                      sum([hash((site, spec)) for site, spec in self.transPairs])

        self.__hashcache__ = sum([hash((site, spec)) for site, spec in self.transPairs])
        # self.__hashcache__ = hash(tuple(self.transPairs))

    def __eq__(self, other):
        if self.SiteSpecs == other.SiteSpecs:
            return True
        return False

    def __hash__(self):
        return self.__hashcache__

    def g(self, crys, g):
        return self.__class__(self.specList, [site.g(crys, g) for site in self.siteList])

    def strRep(self):
        str= ""
        for site, spec in self.SiteSpecs:
            str += "Spec:{}, site:{},{} ".format(spec, site.ci, site.R)
        return str

    def __repr__(self):
        return self.strRep()

    def __str__(self):
        return self.strRep()


class VectorClusterExpansion(object):
    """
    class to expand velocities and rates in vector cluster functions.
    """
    def __init__(self, sup, clusexp, Tclusexp, jumpnetwork, NSpec, Nvac, vacSite, maxorder, maxorderTrans):
        """
        :param sup : clusterSupercell object
        :param clusexp: cluster expansion about a single unit cell.
        :param mobCountList - list of number of each species in the supercell.
        :param vacSite - the site of the vacancy as a clusterSite object. This does not change during the simulation.
        :param maxorder - the maximum order of a cluster in our cluster expansion.
        In this type of simulations, we consider a solid with a single wyckoff set on which atoms are arranged.
        """
        self.chem = 0  # we'll work with a monoatomic basis
        self.sup = sup
        self.Nsites = len(self.sup.mobilepos)
        self.crys = self.sup.crys
        # vacInd will always be the initial state in the transitions that we consider.
        self.clusexp = clusexp
        self.Tclusexp = Tclusexp
        self.maxOrder = maxorder
        self.vacSpec = NSpec - 1
        self.Nvac = Nvac
        self.NSpec = NSpec
        self.mobList = list(range(NSpec))
        self.vacSite = vacSite  # This stays fixed throughout the simulation, so makes sense to store it.
        self.jumpnetwork = jumpnetwork

        start = time.time()
        self.SpecClusters = self.recalcClusters()
        print("Generated clusters with species: {:.4f}".format(time.time()-start))
        start = time.time()
        self.SiteSpecInteractions, self.maxInteractCount = self.generateSiteSpecInteracts()
        # add a small check here - maybe we'll remove this later
        print("Generated Interaction data {:.4f}".format(time.time() - start))
        start = time.time()
        self.vecClus, self.vecVec, self.clus2LenVecClus = self.genVecClustBasis(self.SpecClusters)
        print("Generated vector basis data {:.4f}".format(time.time() - start))
        start = time.time()
        self.IndexClusters()  #  assign integer identifiers to each cluster
        self.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
        self.indexClustertoVecClus()  # Index where in the vector cluster list a cluster is present
        self.indexClustertoSpecClus()  # Index clusters to symmetry groups
        print("Generated Indexing data {:.4f}".format(time.time() - start))

        # Generate the transitions-based data structures - moved to KRAexpander
        # self.ijList, self.dxList, self.clustersOn, self.clustersOff = self.GetTransActiveClusts(self.jumpnetwork)

        # Generate the complete cluster basis including the arrangement of species on sites other than the vacancy site.
        self.KRAexpander = Transitions.KRAExpand(sup, self.chem, jumpnetwork, maxorderTrans, Tclusexp, NSpec, Nvac, vacSite)

    def recalcClusters(self):
        """
        Intended to take in a site based cluster expansion and recalculate the clusters with species in them
        """
        allClusts = set()
        symClusterList = []
        for clSet in self.clusexp:
            for clust in list(clSet):
                Nsites = len(clust.sites)
                occupancies = list(itertools.product(self.mobList, repeat=Nsites))
                for siteOcc in occupancies:
                    # Make the cluster site object
                    ClustSpec = ClusterSpecies(siteOcc, clust.sites)
                    # check if this has already been considered
                    if ClustSpec in allClusts:
                        continue
                    # Check if number of each species in the cluster is okay
                    mobcount = collections.Counter(siteOcc)
                    # Check if the number of vacancies is kept to the allowed number
                    if mobcount[self.vacSpec] > self.Nvac:
                        continue
                    # Otherwise, find all symmetry-grouped counterparts
                    newSymSet = set([ClustSpec.g(self.crys, g) for g in self.crys.G])
                    allClusts.update(newSymSet)
                    newList = list(newSymSet)
                    symClusterList.append(newList)

        return symClusterList

    def genVecClustBasis(self, specClusters):

        vecClustList = []
        vecVecList = []
        clus2LenVecClus = np.zeros(len(specClusters), dtype=int)
        for clListInd, clList in enumerate(specClusters):
            cl0 = clList[0]
            glist0 = []
            for g in self.crys.G:
                if cl0.g(self.crys, g) == cl0:
                    glist0.append(g)

            G0 = sum([g.cartrot for g in glist0])/len(glist0)
            vals, vecs = np.linalg.eig(G0)
            vecs = np.real(vecs)
            vlist = [vecs[:, i]/np.linalg.norm(vecs[:, i]) for i in range(3) if np.isclose(vals[i], 1.0)]
            clus2LenVecClus[clListInd] = len(vlist)

            if clus2LenVecClus[clListInd] == 0:  # If the vector basis is empty, don't consider the cluster
                # vecClustList.append(clList)
                # vecVecList.append([np.zeros(3) for i in range(len(clList))])
                continue

            for v in vlist:
                newClustList = [cl0]
                # The first cluster being the same helps in indexing
                newVecList = [v]
                for g in self.crys.G:
                    cl1 = cl0.g(self.crys, g)
                    if cl1 in newClustList:
                        continue
                    newClustList.append(cl1)
                    newVecList.append(np.dot(g.cartrot, v))

                vecClustList.append(newClustList)
                vecVecList.append(newVecList)

        return vecClustList, vecVecList, clus2LenVecClus

    def indexVclus2Clus(self):

        self.Vclus2Clus = np.zeros(len(self.vecClus), dtype=int)
        self.Clus2VClus = collections.defaultdict(list)
        for cLlistInd, clList in enumerate(self.SpecClusters):
            if self.clus2LenVecClus[cLlistInd] == 0:  # If the vector basis is empty, don't consider the cluster
                self.Clus2VClus[cLlistInd] = []
                continue
            cl0 = clList[0]
            for vClusListInd, vClusList in enumerate(self.vecClus):
                clVec0 = vClusList[0]
                if clVec0 == cl0:
                    self.Vclus2Clus[vClusListInd] = cLlistInd
                    self.Clus2VClus[cLlistInd].append(vClusListInd)

    def indexClustertoVecClus(self):
        """
        For a given cluster, store which vector cluster it belongs to
        """
        self.clust2vecClus = collections.defaultdict(list)
        for clListInd, clList in enumerate(self.SpecClusters):
            if self.clus2LenVecClus[clListInd] == 0:  # If the vector basis is empty, don't consider the cluster
                continue
            vecClusIndList = self.Clus2VClus[clListInd]
            for clust1 in clList:
                for vecClusInd in vecClusIndList:
                    for clust2Ind, clust2 in enumerate(self.vecClus[vecClusInd]):
                        if clust1 == clust2:
                            self.clust2vecClus[clust1].append((vecClusInd, clust2Ind))

    def indexClustertoSpecClus(self):
        """
        For a given cluster, store which vector cluster it belongs to
        """
        self.clust2SpecClus = {}
        for clListInd, clList in enumerate(self.SpecClusters):
            for clustInd, clust in enumerate(clList):
                self.clust2SpecClus[clust] = (clListInd, clustInd)

    def generateSiteSpecInteracts(self):
        """
        generate interactions for every site - for MC moves
        """
        SiteSpecinteractList = collections.defaultdict(list)
        for siteInd in range(self.Nsites):
            # get the cluster site
            ci, R = self.sup.ciR(siteInd)
            clSite = cluster.ClusterSite(ci=ci, R=R)
            # Now, go through all the clusters
            for cl in [cl for clist in self.SpecClusters for cl in clist]:
                for site, spec in cl.SiteSpecs:
                    if site.ci == ci: # In our case we have only one chemistry.
                        Rtrans = R - site.R
                        interactSupInd = tuple([(self.sup.index(site.R+Rtrans, site.ci)[0], spec)
                                                for site, spec in cl.SiteSpecs])
                        SiteSpecinteractList[(clSite, spec)].append([interactSupInd, cl, Rtrans])

        maxinteractions = max([len(lst) for key, lst in SiteSpecinteractList.items()])
        return SiteSpecinteractList, maxinteractions

    def IndexClusters(self):
        """
        Assign a unique integer to each representative cluster. To help identifying them in JIT operations
        """
        allSpCl = [SpCl for SpClList in self.SpecClusters for SpCl in SpClList]
        self.Clus2Num = {}
        self.Num2Clus = {}

        for i, SpCl in enumerate(allSpCl):
            self.Clus2Num[SpCl] = i
            self.Num2Clus[i] = SpCl

    def makeJitInteractionsData(self, Energies):
        """
        Function to represent all the data structures in the form of numpy arrays so that they can be accelerated with
        numba's jit compilations.
        Data structures to cast into numpy arrays:
        SiteInteractions
        KRAexpander.clusterSpeciesJumps - these correspond to transitions - We'll proceed with this later on
        """

        # first, we assign unique integers to interactions
        start = time.time()
        InteractionIndexDict = {}
        siteSortedInteractionIndexDict = {}
        InteractionRepClusDict = {}
        Index2InteractionDict = {}
        repClustCounter = collections.defaultdict(int)
        # siteSpecInteractIndexDict = collections.defaultdict(list)

        # while we're at it, let's also store which siteSpec contains which interact
        numInteractsSiteSpec = np.zeros((self.Nsites, self.NSpec), dtype=int)
        SiteSpecInterArray = np.full((self.Nsites, self.NSpec, self.maxInteractCount), -1, dtype=int)

        count = 0  # to keep a steady count of interactions.
        for key, interactInfoList in self.SiteSpecInteractions.items():
            keySite = self.sup.index(key[0].R, key[0].ci)[0]  # the "index" function applies PBC to sites outside sup.
            keySpec = key[1]
            numInteractsSiteSpec[keySite, keySpec] = len(interactInfoList)
            for interactInd, interactInfo in enumerate(interactInfoList):
                interaction = interactInfo[0]
                if interaction in InteractionIndexDict:
                    #siteSpecInteractIndexDict[(keySite, keySpec)].append(InteractionIndexDict[interaction])
                    SiteSpecInterArray[keySite, keySpec, interactInd] = InteractionIndexDict[interaction]
                    continue
                else:
                    # assign a new unique integer to this interaction
                    InteractionIndexDict[interaction] = count
                    repClustCounter[interactInfo[1]] += 1
                    # also sort the sites by the supercell site indices - will help in identifying TSclusters as interactions
                    # later on
                    InteractionRepClusDict[interaction] = interactInfo[1]
                    Index2InteractionDict[count] = interaction
                    SiteSpecInterArray[keySite, keySpec, interactInd] = count
                    count += 1

        print("Done Indexing interactions : {}".format(time.time() - start))
        # Now that we have integers assigned to all the interactions, let's store their data as numpy arrays
        numInteracts = len(InteractionIndexDict)

        # 1. Store chemical data
        start = time.time()
        # we'll need the number of sites in each interaction
        numSitesInteracts = np.zeros(numInteracts, dtype=int)

        # and the supercell sites in each interaction
        SupSitesInteracts = np.full((numInteracts, self.maxOrder), -1, dtype=int)

        # and the species on the supercell sites in each interaction
        SpecOnInteractSites = np.full((numInteracts, self.maxOrder), -1, dtype=int)

        # and we want to the know the symmetry class of each interaction
        Interact2RepClusArray = np.full(numInteracts, -1, dtype=int)
        Interact2SymClassArray = np.full(numInteracts, -1, dtype=int)

        for (key, interaction) in Index2InteractionDict.items():
            numSitesInteracts[key] = len(interaction)

            # Now get the representative cluster
            repClus = InteractionRepClusDict[interaction]

            # Now get the index assigned to this cluster
            clustInd = self.Clus2Num[repClus]
            Interact2RepClusArray[key] = clustInd

            # Now get the symmetry class for this representative cluster
            (clListInd, clInd) = self.clust2SpecClus[repClus]

            Interact2SymClassArray[key] = clListInd

            for idx, (intSite, intSpec) in enumerate(interaction):
                SupSitesInteracts[key, idx] = intSite
                SpecOnInteractSites[key, idx] = intSpec


        print("Done with chemical and symmetry class data for interactions : {}".format(time.time() - start))

        # 2. Store energy data and vector data
        start = time.time()
        Interaction2En = np.zeros(numInteracts, dtype=float)
        numVecsInteracts = np.full(numInteracts, -1, dtype=int)
        VecsInteracts = np.zeros((numInteracts, 3, 3))
        VecGroupInteracts = np.full((numInteracts, 3), -1, dtype=int)
        for interaction, repClus in InteractionRepClusDict.items():
            idx = InteractionIndexDict[interaction]
            # get the energy index here
            Interaction2En[idx] = Energies[self.clust2SpecClus[repClus][0]]
            # get the vector basis data here
            # if vector basis is empty, keep no of elements to -1.
            if self.clus2LenVecClus[self.clust2SpecClus[repClus][0]] == 0:
                continue
            vecList = self.clust2vecClus[repClus]
            # store the number of vectors in the basis
            numVecsInteracts[idx] = len(vecList)
            # store the vector
            for vecidx, tup in enumerate(vecList):
                VecsInteracts[idx, vecidx, :] = self.vecVec[tup[0]][tup[1]].copy()
                VecGroupInteracts[idx, vecidx] = tup[0]
        print("Done with vector and energy data for interactions : {}".format(time.time() - start))

        vacSiteInd = self.sup.index(self.vacSite.R, self.vacSite.ci)[0]

        return numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, VecsInteracts,\
               VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray, vacSiteInd, InteractionIndexDict, InteractionRepClusDict,\
               Index2InteractionDict, repClustCounter, Interact2RepClusArray, Interact2SymClassArray

    def makeSiteIndToSite(self):
        Nsites = self.Nsites
        N_units = self.sup.superlatt[0, 0]
        siteIndtoR = np.zeros((Nsites, 3), dtype=int)
        RtoSiteInd = np.zeros((N_units, N_units, N_units), dtype=int)

        for siteInd in range(Nsites):
            R = self.sup.ciR(siteInd)[1]
            siteIndtoR[siteInd, :] = R
            RtoSiteInd[R[0], R[1], R[2]] = siteInd
        return siteIndtoR, RtoSiteInd

class MCSamplerClass(object):

    def __init__(self, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,
                 VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray,
                 numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
                 FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng,
                 vacSiteInd, mobOcc):

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, self.numVecsInteracts,\
        self.VecsInteracts, self.VecGroupInteracts, self.numInteractsSiteSpec, self.SiteSpecInterArray, self.vacSiteInd = \
        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,\
        VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray, vacSiteInd

        self.numSitesTSInteracts, self.TSInteractSites, self.TSInteractSpecs =\
            numSitesTSInteracts, TSInteractSites, TSInteractSpecs

        self.jumpFinSites, self.jumpFinSpec, self.FinSiteFinSpecJumpInd, self.numJumpPointGroups, self.numTSInteractsInPtGroups,\
        self.JumpInteracts, self.Jump2KRAEng =\
            jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups,\
            JumpInteracts, Jump2KRAEng

        # check if proper sites and species data are entered
        self.Nsites, self.Nspecs = numInteractsSiteSpec.shape[0], numInteractsSiteSpec.shape[1]
        self.mobOcc = mobOcc

        # generate offsite counts for state interactions
        self.OffSiteCount = np.zeros(len(numSitesInteracts), dtype=int)
        for interactIdx in range(len(numSitesInteracts)):
            numSites = numSitesInteracts[interactIdx]
            for intSiteind in range(numSites):
                interSite = SupSitesInteracts[interactIdx, intSiteind]
                interSpec = SpecOnInteractSites[interactIdx, intSiteind]
                if mobOcc[interSite] != interSpec:
                    self.OffSiteCount[interactIdx] += 1

    def makeMCsweep(self, mobOcc, OffSiteCount, TransOffSiteCount,
                    SwapTrials, beta, randarr, Nswaptrials):

        # TODO : Need to implement biased sampling methods to select sites from TSinteractions with more prob.
        for swapcount in range(Nswaptrials):
            # first select two random sites to swap - for now, let's just select naively.
            siteA = SwapTrials[swapcount, 0]
            siteB = SwapTrials[swapcount, 1]

            specA = mobOcc[siteA]
            specB = mobOcc[siteB]

            delE = 0.
            # Next, switch required sites off
            for interIdx in range(self.numInteractsSiteSpec[siteA, specA]):
                # check if an interaction is on
                interMainInd = self.SiteSpecInterArray[siteA, specA, interIdx]
                # offscount = OffSiteCount[interMainInd]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                OffSiteCount[interMainInd] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, specB]):
                interMainInd = self.SiteSpecInterArray[siteB, specB, interIdx]
                # offscount = OffSiteCount[interMainInd]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                OffSiteCount[interMainInd] += 1

            # Next, switch required sites on
            for interIdx in range(self.numInteractsSiteSpec[siteA, specB]):
                interMainInd = self.SiteSpecInterArray[siteA, specB, interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]

            for interIdx in range(self.numInteractsSiteSpec[siteB, specA]):
                interMainInd = self.SiteSpecInterArray[siteB, specA, interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]

            self.delE = delE  # for testing purposes
            # do the selection test
            if -beta * delE > randarr[swapcount]:
                # swap the sites to get to the next state
                mobOcc[siteA] = specB
                mobOcc[siteB] = specA
                # OffSiteCount is already updated to that of the new state.

            else:
                # revert back the off site counts, because the state has not changed
                for interIdx in range(self.numInteractsSiteSpec[siteA, specA]):
                    # interMainInd = self.SiteSpecInterArray[siteA, specA, interIdx]
                    OffSiteCount[self.SiteSpecInterArray[siteA, specA, interIdx]] -= 1

                for interIdx in range(self.numInteractsSiteSpec[siteB, specB]):
                    # interMainInd = self.SiteSpecInterArray[siteB, specB, interIdx]
                    OffSiteCount[self.SiteSpecInterArray[siteB, specB, interIdx]] -= 1

                for interIdx in range(self.numInteractsSiteSpec[siteA, specB]):
                    # interMainInd = self.SiteSpecInterArray[siteA, specB, interIdx]
                    OffSiteCount[self.SiteSpecInterArray[siteA, specB, interIdx]] += 1

                for interIdx in range(self.numInteractsSiteSpec[siteB, specA]):
                    # interMainInd = self.SiteSpecInterArray[siteB, specA, interIdx]
                    OffSiteCount[self.SiteSpecInterArray[siteB, specA, interIdx]] += 1

        # make the offsite for the transition states
        for TsInteractIdx in range(len(self.TSInteractSites)):
            TransOffSiteCount[TsInteractIdx] = 0
            for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                if mobOcc[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                    TransOffSiteCount[TsInteractIdx] += 1

    def Expand(self, state, ijList, dxList, OffSiteCount, TSOffSiteCount, lenVecClus, beta):

        del_lamb_mat = np.zeros((lenVecClus, lenVecClus, ijList.shape[0]))
        delxDotdelLamb = np.zeros((lenVecClus, ijList.shape[0]))

        ratelist = np.zeros(ijList.shape[0])

        siteA, specA = self.vacSiteInd, self.Nspecs - 1
        # go through all the transitions
        for jumpInd in range(ijList.shape[0]):
            del_lamb = np.zeros((lenVecClus, 3))

            # Get the transition index
            siteB, specB = ijList[jumpInd], state[ijList[jumpInd]]
            transInd = self.FinSiteFinSpecJumpInd[siteB, specB]

            # First, work on getting the KRA energy for the jump
            delEKRA = 0.0
            # We need to go through every point group for this jump
            for tsPtGpInd in range(self.numJumpPointGroups[transInd]):
                for interactInd in range(self.numTSInteractsInPtGroups[transInd, tsPtGpInd]):
                    # Check if this interaction is on
                    interactMainInd = self.JumpInteracts[transInd, tsPtGpInd, interactInd]
                    if TSOffSiteCount[interactMainInd] == 0:
                        delEKRA += self.Jump2KRAEng[transInd, tsPtGpInd, interactInd]

            # next, calculate the energy change due to site swapping

            delE = 0.0
            # Switch required sites off
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
                # check if an interaction is on
                interMainInd = self.SiteSpecInterArray[siteA, state[siteA], interIdx]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                    # take away the vectors for this interaction
                    for i in range(self.numVecsInteracts[interMainInd]):
                        del_lamb[self.VecGroupInteracts[interMainInd, i]] -= self.VecsInteracts[interMainInd, i, :]
                OffSiteCount[interMainInd] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteB, state[siteB], interIdx]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                    for i in range(self.numVecsInteracts[interMainInd]):
                        del_lamb[self.VecGroupInteracts[interMainInd, i]] -= self.VecsInteracts[interMainInd, i, :]
                # OffSiteCount[interMainInd] += 1

            # Next, switch required sites on
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteA, state[siteB], interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]
                    # add the vectors for this interaction
                    for i in range(self.numVecsInteracts[interMainInd]):
                        del_lamb[self.VecGroupInteracts[interMainInd, i]] += self.VecsInteracts[interMainInd, i, :]

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
                interMainInd = self.SiteSpecInterArray[siteB, state[siteA], interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]
                    # add the vectors for this interaction
                    for i in range(self.numVecsInteracts[interMainInd]):
                        del_lamb[self.VecGroupInteracts[interMainInd, i]] += self.VecsInteracts[interMainInd, i, :]

            # Energy change computed, now expand
            ratelist[jumpInd] = np.exp(-(0.5 * delE + delEKRA) * beta)
            del_lamb_mat[:, :, jumpInd] = np.dot(del_lamb, del_lamb.T)

            delxDotdelLamb[:, jumpInd] = np.tensordot(del_lamb, dxList[jumpInd], axes=(1, 0))

            # Next, restore OffSiteCounts to original values for next jump, as well as
            # for use in the next MC sweep.
            # During switch-off operations, offsite counts were increased by one.
            # So decrease them back by one
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
                OffSiteCount[self.SiteSpecInterArray[siteA, state[siteA], interIdx]] -= 1

            # During switch-on operations, offsite counts were decreased by one.
            # So increase them back by one
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
                OffSiteCount[self.SiteSpecInterArray[siteA, state[siteB], interIdx]] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
                OffSiteCount[self.SiteSpecInterArray[siteB, state[siteA], interIdx]] += 1

        ax2 = np.array((0, 2))
        ax3 = np.array((0, 1))
        Wbar = np.tensordot(ratelist, del_lamb_mat, axes=ax2)
        Bbar = np.tensordot(ratelist, delxDotdelLamb, axes=ax3)

        return Wbar, Bbar
                







