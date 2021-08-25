from onsager import crystal, supercell, cluster
import numpy as np
import collections
import itertools
import Transitions
import Cluster_Expansion
import unittest
import time

class testKRA(unittest.TestCase):

    def setUp(self):
        self.NSpec = 3
        self.Nvac = 1
        self.MaxOrder = 3
        self.MaxOrderTrans = 3
        self.crys = crystal.Crystal.BCC(0.2836, chemistry="A")
        self.jnetBCC = self.crys.jumpnetwork(0, 0.26)
        self.superlatt = 8 * np.eye(3, dtype=int)
        self.superBCC = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superBCC.mobilepos)
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superBCC.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.clusexp = cluster.makeclusters(self.crys, 0.284, self.MaxOrder)
        self.Tclusexp = cluster.makeclusters(self.crys, 0.29, self.MaxOrderTrans)
        self.KRAexpander = Transitions.KRAExpand(self.superBCC, 0, self.jnetBCC, self.Tclusexp, self.Tclusexp, self.NSpec,
                                                 self.Nvac, self.vacsite)
        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superBCC, self.clusexp, self.Tclusexp, self.jnetBCC,
                                                                 self.NSpec, self.vacsite, self.MaxOrder,
                                                                 self.MaxOrderTrans)

        print(self.VclusExp.clus2LenVecClus)

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))
        self.KRAEnergies = [np.random.rand(len(val)) for (key, val) in self.VclusExp.KRAexpander.clusterSpeciesJumps.items()]

        self.mobOccs = np.zeros((self.NSpec, numSites), dtype=int)
        for site in range(1, numSites):
            spec = np.random.randint(0, self.NSpec - 1)
            self.mobOccs[spec][site] = 1
        self.mobOccs[-1, self.vacsiteInd] = 1
        self.mobCountList = [np.sum(self.mobOccs[i]) for i in range(self.NSpec)]
        print("Done setting up")

    def test_groupTrans(self):
        """
        To check if the group operations that form the clusters keep the transition sites unchanged.
        """
        for key, clusterLists in self.KRAexpander.SymTransClusters.items():

            ciA, RA = self.superBCC.ciR(key[0])
            ciB, RB = self.superBCC.ciR(key[1])
            siteA = cluster.ClusterSite(ci=ciA, R=RA)
            siteB = cluster.ClusterSite(ci=ciB, R=RB)
            self.assertEqual(siteA, self.vacsite)

            # Get the point group of this transition
            Glist = []
            for g in self.crys.G:
                siteANew = siteA.g(self.crys, g)
                siteBNew = siteB.g(self.crys, g)
                if siteA == siteANew and siteB == siteBNew:
                    Glist.append(g)

            clusterListCount = collections.defaultdict(int)  # each cluster should only appear in one list
            for clist in clusterLists:
                cl0 = clist[0]
                for clust in clist:
                    self.assertTrue(clust.Norder > 0)
                    clusterListCount[clust] += 1
                    count = 0
                    for g in Glist:
                        clNew = cl0.g(self.crys, g)
                        if clNew == clust:
                            count += 1
                            # Check that every group operation which maps the starting cluster to some cluster
                            # in the list keeps the transition unchanged.
                            self.assertEqual(siteA, siteA.g(self.crys, g))
                            self.assertEqual(siteB, siteB.g(self.crys, g))
                    self.assertNotEqual(count, 0)
            for clust, count in clusterListCount.items():
                self.assertEqual(count, 1)

    def test_species_grouping(self):
        """
        The objective for this is to check that each clusterlist is repeated as many times as there should be
        species in its sites.
        """
        # First, count that every transition has every specJ at the end site
        clusterSpeciesJumps = self.KRAexpander.clusterSpeciesJumps

        counter = collections.defaultdict(int)
        for key, items in clusterSpeciesJumps.items():
            counter[(key[0], key[1])] += 1

        for key, item in counter.items():
            self.assertEqual(item, self.NSpec-1)

        # Now check that all possible atomic arrangements have been accounted for
        for key, SpeciesclusterLists in clusterSpeciesJumps.items():
            # check that the initial site is the vacancy site
            self.assertEqual(key[0], self.VclusExp.sup.index(self.vacsite.R, self.vacsite.ci)[0])
            clusterCounts = collections.defaultdict(int)
            for species, clusterList in SpeciesclusterLists:
                cl0 = clusterList[0]
                self.assertEqual(cl0.Norder, len(species))
                self.assertEqual(cl0.Norder+2, len(cl0.sites))
                clusterCounts[cl0] += 1

            for cl0, count in clusterCounts.items():
                numTrue = (self.NSpec-1)**cl0.Norder  # except the vacancy, any other species can be assigned to the sites.
                self.assertEqual(numTrue, count, msg="{}, {}, {}".format(numTrue, count, cl0.Norder))

    def test_KRA(self):
        """
        Checking whether the KRA expansions are done correctly
        """
        # Go through each transition
        for transition, clusterLists in self.KRAexpander.clusterSpeciesJumps.items():
            # get the number of clusterLists, and generate that many coefficients
            KRACoeffs = np.array([np.random.rand() for i in range(len(clusterLists))])
            valOn = np.zeros(len(KRACoeffs))
            # Now go through the clusterLists and note which clusters are on
            for Idx, (tup, clList) in enumerate(clusterLists):
                for cl in clList:
                    # check if this cluster is On
                    prod = 1
                    countVacSite = 0
                    countFinSite = 0
                    for siteInd, site in enumerate(cl.sites):
                        siteIdx = self.superBCC.index(site.R, site.ci)[0]
                        if siteInd == 0:
                            # vacancy site is always occupied by vacancy
                            countVacSite += 1
                            self.assertEqual(siteIdx, self.vacsiteInd)
                            self.assertEqual(self.mobOccs[-1, siteInd], 1)
                        elif siteInd == 1:
                            # SpecJ
                            countFinSite += 1
                            specJ = transition[2]
                            # check if this site is occupied
                            self.assertEqual(self.superBCC.index(site.R, site.ci)[0], transition[1])
                            if self.mobOccs[specJ][transition[1]] == 0:
                                continue  # this is not the transition we are looking for for this state
                        elif self.mobOccs[tup[siteInd - 2], siteIdx] == 0:
                            prod = 0

                    self.assertEqual(countFinSite, 1)
                    self.assertEqual(countVacSite, 1)

                    if prod == 1:
                        valOn[Idx] += KRACoeffs[Idx]

            KRAen = np.sum(valOn)
            KRAcalc = self.KRAexpander.GetKRA(transition, self.mobOccs, KRACoeffs)
            self.assertTrue(np.allclose(KRAen, KRAcalc), msg="{}, {}".format(KRAen, KRAcalc))


class test_Vector_Cluster_Expansion(unittest.TestCase):

    def setUp(self):
        self.NSpec = 3
        self.Nvac = 1
        self.MaxOrder = 3
        self.MaxOrderTrans = 3
        self.crys = crystal.Crystal.BCC(0.2836, chemistry="A")
        self.jnetBCC = self.crys.jumpnetwork(0, 0.26)
        self.superlatt = 8 * np.eye(3, dtype=int)
        self.superBCC = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superBCC.mobilepos)
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superBCC.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.clusexp = cluster.makeclusters(self.crys, 0.284, self.MaxOrder)
        self.Tclusexp = cluster.makeclusters(self.crys, 0.29, self.MaxOrderTrans)
        self.KRAexpander = Transitions.KRAExpand(self.superBCC, 0, self.jnetBCC, self.Tclusexp, self.Tclusexp, self.NSpec,
                                                 self.Nvac, self.vacsite)
        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superBCC, self.clusexp, self.Tclusexp, self.jnetBCC,
                                                                 self.NSpec, self.vacsite, self.MaxOrder,
                                                                 self.MaxOrderTrans)

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))
        self.KRAEnergies = [np.random.rand(len(val)) for (key, val) in self.VclusExp.KRAexpander.clusterSpeciesJumps.items()]

        self.mobOccs = np.zeros((self.NSpec, numSites), dtype=int)
        for site in range(1, numSites):
            spec = np.random.randint(0, self.NSpec - 1)
            self.mobOccs[spec][site] = 1
        self.mobOccs[-1, self.vacsiteInd] = 1
        self.mobCountList = [np.sum(self.mobOccs[i]) for i in range(self.NSpec)]
        print("Done setting up cluster expansion tests.")

    def test_spec_assign(self):

        # check that every cluster appears just once in just one symmetry list
        clList = [cl for clList in self.VclusExp.SpecClusters for cl in clList]

        for cl1 in clList:
            count = 0
            for cl2 in clList:
                if cl1 == cl2:
                    count += 1
            assert count == 1 # check that the cluster occurs only once

        # let's test if the number of symmetric site clusters generated is the same
        sitetuples = set([])
        for clusterList in self.VclusExp.SpecClusters:
            for clust in clusterList:
                Rtrans = sum([site.R for site in clust.siteList]) // len(clust.siteList)
                sites_shift = tuple([site-Rtrans for site in clust.siteList])
                # check correct translations
                sitesBuilt = tuple([site for site, spec in clust.transPairs])
                if self.VclusExp.zeroClusts:
                    self.assertEqual(sites_shift, sitesBuilt)
                sitetuples.add(sites_shift)

        sitesFromClusexp = set([])
        for clSet in self.clusexp:
            for cl in list(clSet):
                for g in self.VclusExp.crys.G:
                    newsitelist = tuple([site.g(self.VclusExp.crys, g) for site in cl.sites])
                    Rtrans = sum([site.R for site in newsitelist])//len(newsitelist)
                    sitesFromClusexp.add(tuple([site - Rtrans for site in newsitelist]))

        total_reps = len(sitesFromClusexp)

        self.assertEqual(total_reps, len(sitetuples), msg="{}, {}".format(total_reps, len(sitetuples)))

        # This test is specifically for a BCC, 3-spec, 3-order, 2nn cluster expansion
        oneCounts = 0
        twocounts = 0
        threecounts = 0
        for siteList in sitesFromClusexp:
            ln=len(siteList)
            if ln == 1:
                oneCounts += 1
            elif ln == 2:
                twocounts += 1
            elif ln == 3:
                threecounts += 1

        self.assertEqual((oneCounts, twocounts, threecounts), (1, 14, 24))

        # Go through all the site clusters:
        total_spec_clusts = 0
        for clSites in sitesFromClusexp:
            ordr = len(clSites)
            # for singletons, any species assignment is acceptable
            if ordr == 1:
                total_spec_clusts += self.NSpec
                continue
            # Now, for two and three body clusters:
            # we can have max one vacancy at a site, and any other site can have any other occupancy.
            # or we can have no vacancy at all in which case it's just (Nspec - 1)**ordr
            # Also, Since we are in BCC lattices, we can symmetrically permute two sites in every two or three body cluster
            total_spec_clusts += (ordr * (self.NSpec - 1)**(ordr - 1) + (self.NSpec-1)**ordr)//2
        #
        total_code = sum([len(lst) for lst in self.VclusExp.SpecClusters])
        #
        self.assertEqual(total_code, total_spec_clusts, msg="{}, {}".format(total_code, total_spec_clusts))

        oneBody = 0
        TwoBody = 0
        ThreeBody = 0

        oneBodyList = []
        TwoBodyList = []
        ThreeBodyList = []
        # Check some of the numbers explicitly - this part is the definitive test
        for clusterList in self.VclusExp.SpecClusters:
            for clust in clusterList:
                order = len(clust.SiteSpecs)
                if order == 1:
                    oneBody += 1
                    oneBodyList.append(clust)
                if order == 2:
                    TwoBody += 1
                    TwoBodyList.append(clust)

                if order == 3:
                    ThreeBody += 1
                    ThreeBodyList.append(clust)

        self.assertEqual(oneBody, 3)
        self.assertEqual(TwoBody, 56)
        self.assertEqual(ThreeBody, 240)

        # Now let's check some two body numbers
        vacCount = 0
        nonVacCount = 0
        for clust in TwoBodyList:
            specList = clust.specList
            if any([spec == 2 for spec in specList]):
                vacCount += 1
            else:
                nonVacCount += 1

        self.assertEqual(vacCount, 2*2*4 + 2*2*3)  # The first term is for nearest neighbor, the second for second nearest neighbor
        self.assertEqual(nonVacCount, 2*2*4 + 2*2*3)

        # Now let's check some three body numbers
        vacCount = 0
        nonVacCount = 0
        for clust in ThreeBodyList:
            specList = clust.specList
            if any([spec == 2 for spec in specList]):
                vacCount += 1
            else:
                nonVacCount += 1

        self.assertEqual(vacCount, (3 * 2 * 2) * 12)
        self.assertEqual(nonVacCount, (2 * 2 * 2) * 12)
        print("Done assignment tests")


    def test_genvecs(self):
        """
        Here, we test if we have generated the vector cluster basis (site-based only) properly
        """
        for clListInd, clList, vecList in zip(itertools.count(), self.VclusExp.vecClus,
                                              self.VclusExp.vecVec):
            self.assertEqual(len(clList), len(vecList))
            cl0, vec0 = clList[0], vecList[0]
            for clust, vec in zip(clList, vecList):
                # First check that symmetry operations are consistent
                count = 0
                for g in self.crys.G:
                    if cl0.g(self.crys, g) == clust:
                        count += 1
                        self.assertTrue(np.allclose(np.dot(g.cartrot, vec0), vec) or
                                        np.allclose(np.dot(g.cartrot, vec0) + vec, np.zeros(3)),
                                        msg="\n{}, {} \n{}, {}\n{}\n{}".format(vec0, vec, cl0, clust,
                                                                               self.crys.lattice, g.cartrot))
                self.assertGreater(count, 0)

    def testcluster2vecClus(self):

        # check that every cluster appears just once in just one symmetry list
        clList = [cl for clList in self.VclusExp.SpecClusters for cl in clList]

        for cl1 in clList:
            count = 0
            for cl2 in clList:
                if cl1 == cl2:
                    count += 1
            self.assertEqual(count, 1)  # check that the cluster occurs only once

        for clListInd, clList in enumerate(self.VclusExp.SpecClusters):
            for clust in clList:
                if self.VclusExp.clus2LenVecClus[clListInd] == 0:
                    self.assertFalse(clust in self.VclusExp.clust2vecClus.keys())
                else:
                    self.assertTrue(self.VclusExp.clus2LenVecClus[clListInd] <= 3)
                    vecList = self.VclusExp.clust2vecClus[clust]
                    self.assertTrue(self.VclusExp.clus2LenVecClus[clListInd] == len(vecList))
                    for tup in vecList:
                        self.assertEqual(clust, self.VclusExp.vecClus[tup[0]][tup[1]])

    def test_indexing(self):
        for vclusListInd, clListInd in enumerate(self.VclusExp.Vclus2Clus):
            cl0 = self.VclusExp.vecClus[vclusListInd][0]
            self.assertEqual(cl0, self.VclusExp.SpecClusters[clListInd][0])

    def test_site_interactions(self):
        # test that every interaction is valid with the given Rtrans provided
        # The key site should be present only once
        interaction2RepClust = {}
        interactCounter = collections.defaultdict(int)
        clust2Interact = collections.defaultdict(list)
        self.assertEqual(len(self.VclusExp.SiteSpecInteractions), self.NSpec*len(self.superBCC.mobilepos))
        for (key, infoList) in self.VclusExp.SiteSpecInteractions.items():
            clSite = key[0]
            sp = key[1]
            # print(infoList[0][0])
            for interactionData in infoList:
                interaction = interactionData[0]
                interactCounter[interaction] += 1
                RepClust = interactionData[1]

                if not interaction in interaction2RepClust:
                    interaction2RepClust[interaction] = {RepClust}
                else:
                    interaction2RepClust[interaction].add(RepClust)
                clust2Interact[RepClust].append(interaction)
                count = 0
                for (site, spec) in interaction:
                    if site == clSite and sp == spec:
                        count += 1
                self.assertEqual(count, 1)

        interactCounter.default_factory = None
        clust2Interact.default_factory = None
        # Check that an interaction has occurred as many times as there are sites in it
        for interact, count in interactCounter.items():
            self.assertEqual(count, len(interact))

        # check that all representative clusters have been translated
        self.assertEqual(len(clust2Interact), sum([len(clList) for clList in self.VclusExp.SpecClusters]),
                         msg="found:{}".format(len(clust2Interact)))

        clAll = [cl for clList in self.VclusExp.SpecClusters for cl in clList]
        for clust in clust2Interact.keys():
            self.assertTrue(clust in clAll)

        # check that each interaction corresponds to a rep cluster properly
        for repClust, interactList in clust2Interact.items():
            for interaction in interactList:
                self.assertEqual(len(interaction2RepClust[interaction]), 1)
                self.assertTrue(repClust in interaction2RepClust[interaction])

    def test_trans_count(self):
        # test that all translations of all representative clusters are considered
        allSpCl = [SpCl for SpClList in self.VclusExp.SpecClusters for SpCl in SpClList]
        for (key, infoList) in self.VclusExp.SiteSpecInteractions.items():
            ci, _ = self.VclusExp.sup.ciR(key[0])
            sp = key[1]
            count = 0
            # For each assigned species, check that all translations of a cluster are considered.
            for SpecClus in allSpCl:
                for (site, spec) in SpecClus.SiteSpecs:
                    if spec == sp and site.ci == ci:
                        count += 1  # a translation of this cluster should exist

            self.assertEqual(count, len(infoList), msg="count {}, stored {}\n{}".format(count, len(infoList), key))

    def testcluster2SpecClus(self):

        for clListInd, clList in enumerate(self.VclusExp.SpecClusters):
            for clustInd, clust in enumerate(clList):
                tup = self.VclusExp.clust2SpecClus[clust]
                self.assertEqual(tup[0], clListInd)
                self.assertEqual(tup[1], clustInd)

        






