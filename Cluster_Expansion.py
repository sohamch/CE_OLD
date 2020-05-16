from onsager import crystal, cluster, supercell
import numpy as np
import collections
import itertools
import Transitions
from numba import jit
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
        self.SiteSpecs = set(self.transPairs)
        self.__hashcache__ = int(np.prod(np.array([hash((site, spec)) for site, spec in self.transPairs]))) +\
                             sum([hash((site, spec)) for site, spec in self.transPairs])

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
    def __init__(self, sup, clusexp, jumpnetwork, mobCountList, vacSite, maxorder):
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
        self.maxOrder = maxorder
        self.mobCountList = mobCountList
        self.vacSpec = len(mobCountList) - 1
        self.mobList = list(range(len(mobCountList)))
        self.vacSite = vacSite  # This stays fixed throughout the simulation, so makes sense to store it.
        self.jumpnetwork = jumpnetwork
        self.ScalarBasis = self.createScalarBasis()
        self.SpecClusters = self.recalcClusters()
        self.SiteSpecInteractions, self.maxInteractCount = self.generateSiteSpecInteracts()
        # add a small check here - maybe we'll remove this later

        self.vecClus, self.vecVec = self.genVecClustBasis(self.SpecClusters)
        self.indexVclus2Clus()
        self.indexClustertoVecClus()
        self.indexClustertoSpecClus()

        # Generate the transitions-based data structures
        self.ijList, self.dxList, self.clustersOn, self.clustersOff = self.GetTransActiveClusts(self.jumpnetwork)

        # Generate the complete cluster basis including the arrangement of species on sites other than the vacancy site.
        self.KRAexpander = Transitions.KRAExpand(sup, self.chem, jumpnetwork, clusexp, mobCountList, vacSite)

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
                    # Check if the number of atoms of a given species does not exceed the total number of atoms of that
                    # species in the solid.
                    if any(j > self.mobCountList[i] for i, j in mobcount.items()):
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
        for clList in specClusters:
            cl0 = clList[0]
            glist0 = []
            for g in self.crys.G:
                if cl0.g(self.crys, g) == cl0:
                    glist0.append(g)

            G0 = sum([g.cartrot for g in glist0])/len(glist0)
            vals, vecs = np.linalg.eig(G0)
            vecs = np.real(vecs)
            vlist = [vecs[:, i]/np.linalg.norm(vecs[:, i]) for i in range(3) if np.isclose(vals[i], 1.0)]

            if len(vlist) == 0:  # For systems with inversion symmetry
                vecClustList.append(clList)
                vecVecList.append([np.zeros(3) for i in range(len(clList))])
                continue

            for v in vlist:
                newClustList = [cl0]
                # The first state being the same helps in indexing
                newVecList = [v]
                for g in self.crys.G:
                    cl1 = cl0.g(self.crys, g)
                    if cl1 in newClustList:
                        continue
                    newClustList.append(cl1)
                    newVecList.append(np.dot(g.cartrot, v))

                vecClustList.append(newClustList)
                vecVecList.append(newVecList)

        return vecClustList, vecVecList

    def createScalarBasis(self):
        """
        Function to add in the species arrangements to the cluster basis functions.
        """
        clusterBasis = []
        for clistInd, clist in enumerate(self.clusexp):
            cl0 = list(clist)[0]  # get the representative cluster
            # cluster.sites is a tuple, which maintains the order of the elements.
            Nmobile = len(cl0.sites)
            arrangemobs = itertools.product(self.mobList, repeat=Nmobile)  # arrange mobile sites on mobile species.
            for tup in arrangemobs:
                mobcount = collections.Counter(tup)
                # Check if the number of atoms of a given species does not exceed the total number of atoms of that
                # species in the solid.
                if any(j > self.mobCountList[i] for i, j in mobcount.items()):
                    continue
                # Each cluster is associated with three vectors
                clusterBasis.append((tup, clistInd))
        return clusterBasis

    def indexVclus2Clus(self):

        self.Vclus2Clus = np.zeros(len(self.vecClus), dtype=int)
        self.Clus2VClus = []
        for cLlistInd, clList in enumerate(self.SpecClusters):
            cl0 = clList[0]
            vecClusts = []
            for vClusListInd, vClusList in enumerate(self.vecClus):
                clVec0 = vClusList[0]
                if clVec0 == cl0:
                    self.Vclus2Clus[vClusListInd] = cLlistInd
                    vecClusts.append(vClusListInd)
            self.Clus2VClus.append(vecClusts)

    def indexClustertoVecClus(self):
        """
        For a given cluster, store which vector cluster it belongs to
        """
        self.clust2vecClus = collections.defaultdict(list)
        for clListInd, clList in enumerate(self.SpecClusters):
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
                self.clust2SpecClus[clust]=(clListInd, clustInd)

    def Expand(self, beta, mobOccs, EnCoeffs, KRACoeffs):

        """
        :param beta : 1/KB*T
        :param mobOccs: the mobile occupancy in the current state
        :param EnCoeffs: energy interaction coefficients in a cluster expansion
        :param KRACoeffs: kinetic energy coefficients - pre-formed for each transition.
        :return: Wbar, Bbar - rate and bias expansions in the cluster basis
        """

        del_lamb_mat = np.zeros((len(self.vecClus), len(self.vecClus), len(self.ijList)))
        delxDotdelLamb = np.zeros((len(self.vecClus), len(self.ijList)))

        # To be tensor dotted with ratelist with axes = (0,1)
        ratelist = np.zeros(len(self.ijList))

        for (jnum, ij, dx) in zip(itertools.count(), self.ijList, self.dxList):
            specJ = sum([occ[ij[1][0]] * label for label, occ in enumerate(mobOccs)])
            del_lamb = np.zeros((len(self.vecClus), 3))
            indB, siteB = ij[1]
            indA, siteA = ij[0]

            if siteA != self.vacSite:
                raise ValueError("Incorrect initial jump site for vacancy. Got {}, expected {}".format(siteA,
                                                                                                       self.vacSite))
            # Get the KRA energy for this jump - stored in a dictionary element with key (indA, indB, specJ)
            delEKRA = self.KRAexpander.GetKRA((indA, indB, specJ), mobOccs, KRACoeffs[(indA, indB, specJ)])
            delE = 0.0  # This will added to the KRA energy to get the activation barrier

            # switch the occupancies in the final state
            # put in small check to see we have calculated specJ correctly
            assert(mobOccs[specJ][indB] == 1)

            mobOccs_final = mobOccs.copy()
            mobOccs_final[-1][indA] = 0
            mobOccs_final[-1][indB] = 1
            mobOccs_final[specJ][indA] = 1
            mobOccs_final[specJ][indB] = 0
            # (1) First, we deal with clusters that need to be switched off
            for clusterTup in self.clustersOff[(self.vacSite, self.vacSpec)] + self.clustersOff[(siteB, specJ)]:
                # Check if the cluster is on
                if all([mobOccs[spec][siteInd] == 1 for siteInd, spec in clusterTup[3]]):
                    del_lamb[clusterTup[0]] -= clusterTup[4]  # take away the vector associated with it.
                    delE -= EnCoeffs[self.Vclus2Clus[clusterTup[0]]]  # take away the energy coefficient

            # (2) Now, we deal with clusters that need to be switched on
            for clusterTup in self.clustersOn[(siteB, self.vacSpec)] + self.clustersOn[(self.vacSite, specJ)]:
                # Check if the cluster is on
                if all([mobOccs_final[spec][siteInd] == 1 for siteInd, spec in clusterTup[3]]):
                    del_lamb[clusterTup[0]] += clusterTup[4]  # add the vector associated with it.
                    delE += EnCoeffs[self.Vclus2Clus[clusterTup[0]]]  # add the energy coefficient

            # append to the rateList
            ratelist[jnum] = np.exp(-(0.5*delE + delEKRA)*beta)

            # Create the matrix to find Wbar
            del_lamb_mat[:, :, jnum] = np.dot(del_lamb, del_lamb.T)

            # Create the matrix to find Bbar
            delxDotdelLamb[:, jnum] = np.tensordot(del_lamb, dx, axes=(1, 0))

        Wbar = np.tensordot(ratelist, del_lamb_mat, axes=(0, 2))
        Bbar = np.tensordot(ratelist, delxDotdelLamb, axes=(0, 1))

        return Wbar, Bbar

    def GetTransActiveClusts(self, jumpnetwork):
        """
        Function to find and store those clusters that wither turn on or off due to transitons
        out of a given state
        :param jumpnetwork: vacancy jumpnetwork
        """
        ijList = []
        dxList = []
        clusterTransOff = collections.defaultdict(list)  # these are clusters that need to be turned off
        clusterTransOn = collections.defaultdict(list)  # these are clusters that need to be turned off

        # pre-process those clusters that contain vacancy at vacSite.
        for vclusListInd, clustList, vecList in zip(itertools.count(), self.vecClus, self.vecVec):
            for clInd, clust, vec in zip(itertools.count(), clustList, vecList):
                for site, spec in clust.SiteSpecs:
                    # see if the cluster has a vacancy site with the vacancy on it
                    if site.ci == self.vacSite.ci:
                        Rt = self.vacSite.R - site.R
                        transSites = tuple((site + Rt, spec) for site, spec in clust.SiteSpecs)
                        transSiteInds = tuple((self.sup.index(site.R, site.ci)[0], spec) for site, spec in transSites)
                        if spec == self.vacSpec:
                            clusterTransOff[(self.vacSite, self.vacSpec)].append([vclusListInd, clInd,
                                                                                  transSites, transSiteInds,
                                                                                  vec, Rt])
                        else:
                            # There is some other species at the vacancy site, must be the final state of some jump.
                            clusterTransOn[(self.vacSite, spec)].append([vclusListInd, clInd,
                                                                         transSites, transSiteInds,
                                                                         vec, Rt])

        for jump in [jmp for jList in jumpnetwork for jmp in jList]:
            siteA = cluster.ClusterSite(ci=(self.chem, jump[0][0]), R=np.zeros(3, dtype=int))
            if siteA != self.vacSite:
                # if the initial site of the vacancy is not the vacSite, it is not a jump out of this state.
                # Ignore it - removes reverse jumps from multi-site, single-Wyckoff lattices.
                continue
            Rj, (c, cj) = self.crys.cart2pos(jump[1] -
                                             np.dot(self.crys.lattice, self.crys.basis[self.chem][jump[0][1]]) +
                                             np.dot(self.crys.lattice, self.crys.basis[self.chem][jump[0][0]]))
            # check we have the correct site
            if not cj == jump[0][1]:
                raise ValueError("improper coordinate transformation, did not get same final jump site")
            siteB = cluster.ClusterSite(ci=(self.chem, jump[0][1]), R=Rj)

            indA = self.sup.index(siteA.R, siteA.ci)[0]
            indB = self.sup.index(siteB.R, siteB.ci)[0]
            ijList.append(((indA, siteA), (indB, siteB)))
            dxList.append(jump[1])
            # See which clusters contain specJ at siteJ
            for vclusListInd, clustList, vecList in zip(itertools.count(), self.vecClus, self.vecVec):
                for clInd, clust, vec in zip(itertools.count(), clustList, vecList):
                    for site, spec in clust.SiteSpecs:
                        # see if the cluster has a vacancy site with the vacancy on it
                        if site.ci == siteB.ci:
                            Rt = siteB.R - site.R  # to make the sites coincide
                            transSites = tuple((site + Rt, spec) for (site, spec) in clust.SiteSpecs)
                            transSiteInds = tuple((self.sup.index(site.R, site.ci)[0], spec)
                                                  for site, spec in transSites)
                            if spec != self.vacSpec:
                                # translate all the sites and see if the vacancy is in there as well
                                # if yes, We will not consider this to prevent double counting
                                if not (self.vacSite, self.vacSpec) in transSites:
                                    clusterTransOff[(siteB, spec)].append([vclusListInd, clInd,
                                                                           transSites, transSiteInds,
                                                                           vec, Rt])
                            else:
                                # Check for double counting
                                if self.vacSite not in [site + Rt for site, spec in clust.SiteSpecs]:
                                    # if vacSite IS present, then this means that there is some other species
                                    # on it, which has already been accounted for previously.
                                    clusterTransOn[(siteB, self.vacSpec)].append([vclusListInd, clInd,
                                                                                  transSites, transSiteInds,
                                                                                  vec, Rt])

        return ijList, dxList, clusterTransOn, clusterTransOff

    def generateSiteSpecInteracts(self):
        """
        generate interactions for every site - for MC moves
        """
        SiteSpecinteractList = {}
        InteractCounts = []  # this is to later find out the maximum number of interactions.
        interactionSupIndSet = set()
        for siteInd in range(self.Nsites):
            # get the cluster site
            ci, R = self.sup.ciR(siteInd)
            clSite = cluster.ClusterSite(ci=ci, R=R)
            # assign species to this
            for sp in range(len(self.mobCountList)):
                # For each (clSite, sp) pair, we need an interaction list
                interactionList = []
                # Now, go through all the clusters
                for clListInd, clList in enumerate(self.SpecClusters):
                    for cl in clList:
                        for site, spec in cl.SiteSpecs:
                            if site.ci == clSite.ci and sp == spec:
                                Rtrans = clSite.R - site.R
                                interaction = tuple([(site + Rtrans, spec) for site, spec in cl.SiteSpecs])
                                interactSupInd = tuple([(self.sup.index(site.R, site.ci)[0], spec)
                                                        for site, spec in interaction])
                                # this check is to account for periodic boundary conditions
                                if interactSupInd in interactionSupIndSet:
                                    continue
                                interactionSupIndSet.add(interactSupInd)
                                interactionList.append([interaction, cl, Rtrans])

                SiteSpecinteractList[(clSite, sp)] = interactionList
                InteractCounts.append(len(interactionList))
        maxinteractions = max(InteractCounts)
        return SiteSpecinteractList, maxinteractions

    def makeJitInteractionsData(self, Energies, KRAEnergies):
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
        # siteSpecInteractIndexDict = collections.defaultdict(list)

        # while we're at it, let's also store which siteSpec contains which interact
        numInteractsSiteSpec = np.zeros((self.Nsites, len(self.mobCountList)), dtype=int)
        SiteSpecInterArray = np.full((self.Nsites, len(self.mobCountList), self.maxInteractCount), -1, dtype=int)

        count = 0  # to keep a steady count of interactions.
        for key, interactInfoList in self.SiteSpecInteractions.items():
            keySite = self.sup.index(key[0].R, key[0].ci)[0]  # the "incell" function applies PBC to sites outside sup.
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
                    # also sort the sites by the supercell site indices - will help in identifying TSclusters as interactions
                    # later on
                    InteractSup = tuple([(self.sup.index(clsite.R, clsite.ci)[0], sp)
                                             for (clsite, sp) in interaction])
                    Interact_sort = tuple(sorted(InteractSup, key=lambda x: x[0]))
                    siteSortedInteractionIndexDict[Interact_sort] = count


                    InteractionRepClusDict[interaction] = interactInfo[1]
                    Index2InteractionDict[count] = interaction
                    SiteSpecInterArray[keySite, keySpec, interactInd] = count
                    #siteSpecInteractIndexDict[(keySite, keySpec)].append(count)
                    count += 1

        # check each interaction has its own unique index
        assert len(InteractionIndexDict) == len(Index2InteractionDict)
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

        for (key, interaction) in Index2InteractionDict.items():
            numSitesInteracts[key] = len(interaction)
            for idx, (intSite, intSpec) in enumerate(interaction):
                supSite = self.sup.index(intSite.R, intSite.ci)[0]  # get the supercell site index
                SupSitesInteracts[key, idx] = supSite
                SpecOnInteractSites[key, idx] = intSpec

        print("Done with chemical data for interactions : {}".format(time.time() - start))

        # 2. Store energy data and vector data
        start = time.time()
        Interaction2En = np.zeros(numInteracts)
        numVecsInteracts = np.full(numInteracts, -1, dtype=int)
        VecsInteracts = np.zeros((numInteracts, 3, 3))
        for interaction, repClus in InteractionRepClusDict.items():
            idx = InteractionIndexDict[interaction]
            # get the energy index here
            Interaction2En[idx] = Energies[self.clust2SpecClus[repClus][0]]

            # get the vector basis data here
            vecList = self.clust2vecClus[repClus]
            # store the number of vectors in the basis
            numVecsInteracts[idx] = len(vecList)
            # store the vector
            for vecidx, tup in enumerate(vecList):
                VecsInteracts[idx, vecidx, :] = self.vecVec[tup[0]][tup[1]]
        print("Done with vector and energy data for interactions : {}".format(time.time() - start))

        # 3. Now, we deal with transition state data.
        start = time.time()
        # first, we indexify the transitions
        # See Line 43 of Transition.py - if a jump is not out of vacsite, it is not considered anyway

        # First, we indexify the transition state (site, spec) clusters

        # 3.1 Get the maximum of cluster groups amongst all jumps
        maxInteractGroups = max([len(interactGroupList)
                                 for Jumpkey, interactGroupList in self.KRAexpander.clusterSpeciesJumps.items()])

        # 3.2 get the maximum number of clusters in any given group
        maxInteractsInGroups = max([len(interactGroup[1])
                                     for Jumpkey, interactGroupList in self.KRAexpander.clusterSpeciesJumps.items()
                                     for interactGroup in interactGroupList])

        vacSpecInd = len(self.mobCountList) - 1

        # 3.3 create arrays to
        # store initial and final sites in transitions
        jumpFinSites = np.full(len(self.KRAexpander.clusterSpeciesJumps), -1, dtype=int)
        jumpFinSpec = np.full(len(self.KRAexpander.clusterSpeciesJumps), -1, dtype=int)

        # To store the number of TSInteraction groups for each transition
        numJumpPointGroups = np.full(len(self.KRAexpander.clusterSpeciesJumps), -1, dtype=int)

        # To store the number of clusters in each TSInteraction group for each transition
        numTSInteractsInPtGroups = np.full((len(self.KRAexpander.clusterSpeciesJumps), maxInteractGroups), -1,
                                           dtype=int)

        # To store the main interaction index of each TSInteraction (will be used to check on or off status)
        JumpInteracts = np.full((len(self.KRAexpander.clusterSpeciesJumps), maxInteractGroups, maxInteractsInGroups),
                                -1, dtype=int)

        # To store the KRA energies for each transition state cluster
        Jump2KRAEng = np.zeros((len(self.KRAexpander.clusterSpeciesJumps), maxInteractGroups, maxInteractsInGroups))
        print("Processing {} jumps".format(len(self.KRAexpander.clusterSpeciesJumps)))
        for jumpInd, (Jumpkey, interactGroupList) in zip(itertools.count(),
                                                         self.KRAexpander.clusterSpeciesJumps.items()):

            jumpFinSites[jumpInd] = Jumpkey[1]
            jumpFinSpec[jumpInd] = Jumpkey[2]
            numJumpPointGroups[jumpInd] = len(interactGroupList)

            for interactGroupInd, interactGroup in enumerate(interactGroupList):
                specList = [vacSpecInd, Jumpkey[2]]  # the initial species is the vacancy, and specJ is stored as the key
                spectup, clusterList = interactGroup[0], interactGroup[1]
                for spec in spectup:
                    specList.append(spec)
                # small failsafe
                # if not len(specList) == len(clusterList[0].sites):
                #     print(specList)
                #     print(clusterList[0].sites)
                # assert len(specList) == len(clusterList[0].sites)
                numTSInteractsInPtGroups[jumpInd, interactGroupInd] = len(clusterList)

                for interactInd, TSclust in enumerate(clusterList):
                    TSInteract = tuple([(self.sup.index(clsite.R, clsite.ci)[0], sp)
                                 for clsite, sp in zip(TSclust.sites, specList)])

                    # get the ID of this interaction
                    # sort with the supercell site indices as the keys as each site can appear only once in an
                    # interaction
                    TSInteract_sort = tuple(sorted(TSInteract, key=lambda x: x[0]))
                    TSMainInd = siteSortedInteractionIndexDict[TSInteract_sort]

                    JumpInteracts[jumpInd, interactGroupInd, interactInd] = TSMainInd
                    Jump2KRAEng[jumpInd, interactGroupInd, interactInd] = KRAEnergies[jumpInd][interactGroupInd]

                    # A small check to see that the TSClusters have the sites arranged properly
                    # assert TSInteract[0][0] == self.sup.index(self.vacSite.R, self.vacSite.ci)[0] == Jumpkey[0]
                    # assert TSInteract[1][0] == Jumpkey[1]

        print("Done with transition data : {}".format(time.time() - start))
        vacSiteInd = self.sup.index(self.vacSite.R, self.vacSite.ci)[0]

        return numSitesInteracts, SupSitesInteracts, SpecOnInteractSites,\
               Interaction2En, numVecsInteracts, VecsInteracts, numInteractsSiteSpec,SiteSpecInterArray,\
               jumpFinSites, jumpFinSpec, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng,\
               vacSiteInd, InteractionIndexDict, InteractionRepClusDict, Index2InteractionDict

class MCSampler(object):

    def __init__(self, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,
                 VecsInteracts, numInteractsSiteSpec, SiteSpecInterArray, jumpFinSites, jumpFinSpec,
                 numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, vacSiteInd, Nspec, mobOcc):

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, self.numVecsInteracts,\
        self.VecsInteracts, self.numInteractsSiteSpec, self.SiteSpecInterArray, self.jumpFinSites, self.jumpFinSpec,\
        self.numJumpPointGroups, self.numTSInteractsInPtGroups, self.JumpInteracts, self.Jump2KRAEng, self.vacSiteInd = \
        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,\
        VecsInteracts, numInteractsSiteSpec, SiteSpecInterArray, jumpFinSites, jumpFinSpec,\
        numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng, vacSiteInd

        # check if proper sites and species data are entered
        self.Nsites, self.Nspecs = numInteractsSiteSpec.shape[0], numInteractsSiteSpec.shape[1]

        if self.Nsites*self.Nspecs != len(mobOcc) * Nspec:
            raise ValueError("Inconsistent number of species and sites")

        self.mobOcc = mobOcc
        self.OffSiteCount = np.zeros(len(numSitesInteracts), dtype=int)
        for interactIdx in range(len(numSitesInteracts)):
            numSites = numSitesInteracts[interactIdx]
            for intSiteind in range(numSites):
                interSite = SupSitesInteracts[interactIdx, intSiteind]
                interSpec = SpecOnInteractSites[interactIdx, intSiteind]
                if mobOcc[interSite] != interSpec:
                    self.OffSiteCount[interactIdx] += 1

    def makeMCsweep(self, NswapTrials, beta, Energies):
        """
        This is the function that will do the MC sweeps
        :param NswapTrials: the number of site swaps needed to be done in a single MC sweep
        :param beta : 1/(KB*T)
        update the mobile occupance array and the OffSiteCounts for the MC sweeps
        """
        # TODO : Need to implement biased sampling methods to select sites from TSinteractions with more prob.
        mobOcc = self.mobOcc.copy()

        OffSiteCountOld = self.OffSiteCount.copy()
        OffSiteCountNew = self.OffSiteCount.copy()
        
        randarr = np.random.rand(NswapTrials)

        count = 0
        while count < NswapTrials:
            # first select two random sites to swap - for now, let's just select naively.
            siteA = np.random.randint(0, self.Nsites)
            siteB = np.random.randint(0, self.Nsites)
            if siteA == siteB:
                continue

            delE = 0.
            # Next, switch required sites off
            for interIdx in range(self.numInteractsSiteSpec[siteA, mobOcc[siteA]]):
                # check if an interaction is on
                if OffSiteCountOld[self.SiteSpecInterArray[siteA, mobOcc[siteA], interIdx]] == 0:
                    delE -= Energies[self.Interaction2En[self.SiteSpecInterArray[siteA, mobOcc[siteA], interIdx]]]                
                OffSiteCountNew[self.SiteSpecInterArray[siteA, mobOcc[siteA], interIdx]] += 1
                
            for interIdx in range(self.numInteractsSiteSpec[siteB, mobOcc[siteB]]):
                if OffSiteCountOld[self.SiteSpecInterArray[siteB, mobOcc[siteB], interIdx]] == 0:
                    delE -= Energies[self.Interaction2En[self.SiteSpecInterArray[siteB, mobOcc[siteB], interIdx]]]
                OffSiteCountNew[self.SiteSpecInterArray[siteB, mobOcc[siteB], interIdx]] += 1

            # Next, switch required sites on
            for interIdx in range(self.numInteractsSiteSpec[siteA, mobOcc[siteB]]):
                OffSiteCountNew[self.SiteSpecInterArray[siteA, mobOcc[siteB], interIdx]] -= 1
                if OffSiteCountNew[self.SiteSpecInterArray[siteA, mobOcc[siteB], interIdx]] == 0:
                    delE += Energies[self.Interaction2En[self.SiteSpecInterArray[siteB, mobOcc[siteB], interIdx]]]

            for interIdx in range(self.numInteractsSiteSpec[siteB, mobOcc[siteA]]):
                OffSiteCountNew[self.SiteSpecInterArray[siteB, mobOcc[siteA], interIdx]] -= 1
                if OffSiteCountNew[self.SiteSpecInterArray[siteB, mobOcc[siteA], interIdx]] == 0:
                    delE += Energies[self.Interaction2En[self.SiteSpecInterArray[siteB, mobOcc[siteB], interIdx]]]

            # do the selection test
            if np.exp(-beta*delE) > randarr[count]:
                # update the off site counts
                OffSiteCountOld = OffSiteCountNew.copy()
                # swap the sites to get to the next state
                temp = mobOcc[siteA]
                mobOcc[siteA] = mobOcc[siteB]
                mobOcc[siteB] = temp
            else:
                # revert back the off site counts, because the state has not changed
                OffSiteCountNew = OffSiteCountOld.copy()
                    
            count += 1

        return mobOcc, OffSiteCountNew
                







