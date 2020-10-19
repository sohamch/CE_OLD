import numpy as np
from numba.experimental import jitclass
from numba import int64, float64

# Paste all the function definitions here as comments

np.seterr(all='raise')
MonteCarloSamplerSpec = [
    ("numInteractsSiteSpec", int64[:, :]),
    ("SiteSpecInterArray", int64[:, :, :]),
    ("numSitesInteracts", int64[:]),
    ("numSitesTSInteracts", int64[:]),
    ("SupSitesInteracts", int64[:, :]),
    ("TSInteractSites", int64[:, :]),
    ("TSInteractSpecs", int64[:, :]),
    ("SpecOnInteractSites", int64[:, :]),
    ("Interaction2En", float64[:]),
    ("InteractToRepClus", int64[:]),
    ("numVecsInteracts", int64[:]),
    ("VecsInteracts", float64[:, :, :]),
    ("jumpFinSites", int64[:]),
    ("jumpFinSpec", int64[:]),
    ("numJumpPointGroups", int64[:]),
    ("numTSInteractsInPtGroups", int64[:, :]),
    ("JumpInteracts", int64[:, :, :]),
    ("Jump2KRAEng", float64[:, :, :]),
    ("mobOcc", int64[:]),
    ("vacSiteInd", int64),
    ("Nsites", int64),
    ("Nspecs", int64),
    ("OffSiteCount", int64[:]),
    ("delEArray", float64[:]),
    ("FinSiteFinSpecJumpInd", int64[:, :]),
    ("VecGroupInteracts", int64[:, :])

]


@jitclass(MonteCarloSamplerSpec)
class MCSamplerClass(object):

    def __init__(self, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, InteractToRepClus, numVecsInteracts,
                 VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray,
                 numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
                 FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng,
                 vacSiteInd, mobOcc, OffSiteCount):

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, self.InteractToRepClus,\
        self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts, self.numInteractsSiteSpec, self.SiteSpecInterArray,\
        self.vacSiteInd = \
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, InteractToRepClus, numVecsInteracts, \
            VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray, vacSiteInd

        self.numSitesTSInteracts, self.TSInteractSites, self.TSInteractSpecs = \
            numSitesTSInteracts, TSInteractSites, TSInteractSpecs

        self.jumpFinSites, self.jumpFinSpec, self.FinSiteFinSpecJumpInd, self.numJumpPointGroups, self.numTSInteractsInPtGroups, \
        self.JumpInteracts, self.Jump2KRAEng = \
            jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, \
            JumpInteracts, Jump2KRAEng

        # check if proper sites and species data are entered
        self.Nsites, self.Nspecs = numInteractsSiteSpec.shape[0], numInteractsSiteSpec.shape[1]
        self.mobOcc = mobOcc
        self.OffSiteCount = OffSiteCount.copy()
        for interactIdx in range(numSitesInteracts.shape[0]):
            numSites = numSitesInteracts[interactIdx]
            for intSiteind in range(numSites):
                interSite = SupSitesInteracts[interactIdx, intSiteind]
                interSpec = SpecOnInteractSites[interactIdx, intSiteind]
                if mobOcc[interSite] != interSpec:
                    self.OffSiteCount[interactIdx] += 1

    def makeRepClusOnCounter(self, offsc):
        totRepClus = np.max(self.InteractToRepClus) + 1
        repclusOncounter = np.zeros(totRepClus, dtype=int64)
        for interactInd in range(offsc.shape[0]):
            repClusInd = self.InteractToRepClus[interactInd]
            if offsc[interactInd] == 0:
                repclusOncounter[repClusInd] += 1
        return repclusOncounter

    def makeOffSiteCount(self, state):
        OffSiteCount = np.zeros(self.numSitesInteracts.shape[0], dtype=int64)
        for interactIdx in range(self.numSitesInteracts.shape[0]):
            for intSiteind in range(self.numSitesInteracts[interactIdx]):
                if state[self.SupSitesInteracts[interactIdx, intSiteind]] != self.SpecOnInteractSites[interactIdx, intSiteind]:
                    OffSiteCount[interactIdx] += 1
        return OffSiteCount

    def GetTSOffSite(self, state):
        TransOffSiteCount = np.zeros(self.numSitesTSInteracts.shape[0], dtype=int64)
        for TsInteractIdx in range(self.numSitesTSInteracts.shape[0]):
            for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                if state[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                    TransOffSiteCount[TsInteractIdx] += 1
        return TransOffSiteCount

    def makeMCsweep(self, mobOcc, OffSiteCount, TransOffSiteCount,
                    SwapTrials, beta, randarr, Nswaptrials, vacSiteInd=0):

        # modifying to make representative cluster counts.
        acceptCount = 0
        acceptInd = np.zeros(Nswaptrials, dtype=int64)
        badTrials = 0
        self.delEArray = np.zeros(Nswaptrials)

        Nsites = len(mobOcc)

        count = 0  # to keep a steady count of accepted moves
        swapcount = 0
        while swapcount < Nswaptrials:
            # first select two random sites to swap - for now, let's just select naively.
            siteA = np.random.randint(0, Nsites)
            siteB = np.random.randint(0, Nsites)

            specA = mobOcc[siteA]
            specB = mobOcc[siteB]

            if specA == specB or siteA == vacSiteInd or siteB == vacSiteInd:
                badTrials += 1
                continue

            # If the move is not a bad one, then store it for testing later on
            SwapTrials[swapcount, 0] = siteA
            SwapTrials[swapcount, 1] = siteB

            delE = 0.
            # Next, switch required sites off
            for interIdx in range(self.numInteractsSiteSpec[siteA, specA]):
                # check if an interaction is on
                interMainInd = self.SiteSpecInterArray[siteA, specA, interIdx]
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

            self.delEArray[swapcount] = delE

            # do the selection test
            if -beta*delE > randarr[swapcount]:
                # swap the sites to get to the next state
                mobOcc[siteA] = specB
                mobOcc[siteB] = specA
                # OffSiteCount is already updated to that of the new state.
                acceptCount += 1
                count += 1
                acceptInd[swapcount] = count

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

            swapcount += 1

        # make the offsite for the transition states
        for TsInteractIdx in range(len(self.TSInteractSites)):
            TransOffSiteCount[TsInteractIdx] = 0
            for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                if mobOcc[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                    TransOffSiteCount[TsInteractIdx] += 1

        return acceptCount, badTrials, acceptInd

    def MultiSwapMC(self, mobOcc, OffSiteCount, TransOffSiteCount,
                    SwapTrials, Nswaptrials, beta, randlog, vacSiteInd=0):

        Nsites = len(mobOcc)
        EnChange = 0.
        swapcount = 0

        while swapcount < Nswaptrials:
            # first select two random sites to swap - for now, let's just select naively.
            siteA = np.random.randint(0, Nsites)
            siteB = np.random.randint(0, Nsites)

            specA = mobOcc[siteA]
            specB = mobOcc[siteB]

            if specA == specB or siteA == vacSiteInd or siteB == vacSiteInd:
                continue

            SwapTrials[swapcount, 0] = siteA
            SwapTrials[swapcount, 1] = siteB

            delE = 0.
            # Next, switch required sites off
            for interIdx in range(self.numInteractsSiteSpec[siteA, specA]):
                # check if an interaction is on
                interMainInd = self.SiteSpecInterArray[siteA, specA, interIdx]
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

            # do the selection test
            # swap the sites to get to the next state
            mobOcc[siteA] = specB
            mobOcc[siteB] = specA
            # add the energy to get the energy of the next state
            EnChange += delE
            swapcount += 1

        # Do the Metropolis test

        if -beta*EnChange < randlog:  # Then the whole thing needs to be reverted

            # We need to reverse everything in the exact opposite sequence of how it was produced
            for i in range(Nswaptrials-1, -1, -1):
                # Get the sites that were swapped
                siteA = SwapTrials[i, 0]
                siteB = SwapTrials[i, 1]

                # remember that the sites have been swapped.
                # So, The previous specA is now at siteB
                specA = mobOcc[siteB]
                specB = mobOcc[siteA]

                # revert off site counts
                for interIdx in range(self.numInteractsSiteSpec[siteA, specA]):
                    OffSiteCount[self.SiteSpecInterArray[siteA, specA, interIdx]] -= 1

                for interIdx in range(self.numInteractsSiteSpec[siteB, specB]):
                    OffSiteCount[self.SiteSpecInterArray[siteB, specB, interIdx]] -= 1

                for interIdx in range(self.numInteractsSiteSpec[siteA, specB]):
                    OffSiteCount[self.SiteSpecInterArray[siteA, specB, interIdx]] += 1

                for interIdx in range(self.numInteractsSiteSpec[siteB, specA]):
                    OffSiteCount[self.SiteSpecInterArray[siteB, specA, interIdx]] += 1

                # revert state
                mobOcc[siteA] = specA
                mobOcc[siteB] = specB

        # once the final state is decided, compute the TS energies.
        for TsInteractIdx in range(len(self.TSInteractSites)):
            TransOffSiteCount[TsInteractIdx] = 0
            for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                if mobOcc[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                    TransOffSiteCount[TsInteractIdx] += 1

        return EnChange  # For testing

    def Expand(self, state, ijList, dxList, OffSiteCount, TSOffSiteCount, lenVecClus, beta):

        del_lamb_mat = np.zeros((lenVecClus, lenVecClus, ijList.shape[0]))
        delxDotdelLamb = np.zeros((lenVecClus, ijList.shape[0]))

        ratelist = np.zeros(ijList.shape[0])

        siteA, specA = self.vacSiteInd, self.Nspecs - 1
        # go through all the transition

        for jumpInd in range(ijList.shape[0]):
            del_lamb = np.zeros((lenVecClus, 3))

            # Get the transition index
            siteB, specB = ijList[jumpInd], state[ijList[jumpInd]]
            transInd = self.FinSiteFinSpecJumpInd[siteB, specB]

            # SiteTransArray[jumpInd] = siteB
            # SpecTransArray[jumpInd] = specB

            # First, work on getting the KRA energy for the jump
            delEKRA = 0.0
            # We need to go through every point group for this jump
            for tsPtGpInd in range(self.numJumpPointGroups[transInd]):
                for interactInd in range(self.numTSInteractsInPtGroups[transInd, tsPtGpInd]):
                    # Check if this interaction is on
                    interactMainInd = self.JumpInteracts[transInd, tsPtGpInd, interactInd]
                    if TSOffSiteCount[interactMainInd] == 0:
                        delEKRA += self.Jump2KRAEng[transInd, tsPtGpInd, interactInd]

            # delEKRAarray[jumpInd] = delEKRA

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
                OffSiteCount[interMainInd] += 1

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
                    # for interactions with zero vector basis, numVecsInteracts[interMainInd] = -1 and the
                    # loop doesn't run
                    for i in range(self.numVecsInteracts[interMainInd]):
                        del_lamb[self.VecGroupInteracts[interMainInd, i]] += self.VecsInteracts[interMainInd, i, :]

            # delEarray[jumpInd] = delE
            # Energy change computed, now expand
            # if 0.5 * delE + delEKRA < 0:
            #     print(delE, delEKRA)
            #     raise ValueError("negative activation barrier")
            ratelist[jumpInd] = np.exp(-(0.5 * delE + delEKRA) * beta)
            del_lamb_mat[:, :, jumpInd] = np.dot(del_lamb, del_lamb.T)

            # delxDotdelLamb[:, jumpInd] = np.tensordot(del_lamb, dxList[jumpInd], axes=(1, 0))
            # let's do the tensordot by hand (work on finding numba support for this)
            for i in range(lenVecClus):
                # replace innder loop with outer product
                delxDotdelLamb[i, jumpInd] = np.dot(del_lamb[i, :], dxList[jumpInd, :])

            # Next, restore OffSiteCounts to original values for next jump, as well as
            # for use in the next MC sweep.
            # During switch-off operations, offsite counts were increased by one.
            # So decrease them back by one
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
                OffSiteCount[self.SiteSpecInterArray[siteA, state[siteA], interIdx]] -= 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
                OffSiteCount[self.SiteSpecInterArray[siteB, state[siteB], interIdx]] -= 1

            # During switch-on operations, offsite counts were decreased by one.
            # So increase them back by one
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
                OffSiteCount[self.SiteSpecInterArray[siteA, state[siteB], interIdx]] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
                OffSiteCount[self.SiteSpecInterArray[siteB, state[siteA], interIdx]] += 1

        # Wbar = np.tensordot(ratelist, del_lamb_mat, axes=(0, 2))
        WBar = np.zeros((lenVecClus, lenVecClus))
        for i in range(lenVecClus):
            for j in range(lenVecClus):
                WBar[i, j] += np.dot(del_lamb_mat[i, j, :], ratelist)

        # assert np.allclose(Wbar, WBar)

        # Bbar = np.tensordot(ratelist, delxDotdelLamb, axes=(0, 1))
        BBar = np.zeros(lenVecClus)
        for i in range(lenVecClus):
            BBar[i] = np.dot(ratelist, delxDotdelLamb[i, :])

        # assert np.allclose(Bbar, BBar)

        return WBar, BBar

    def GetNewRandState(self, mobOcc, OffSiteCount, Energy, SwapTrials, Nswaptrials):

        En = Energy
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

            # do the selection test
            # swap the sites to get to the next state
            mobOcc[siteA] = specB
            mobOcc[siteB] = specA
            # add the energy to get the energy of the next state
            En += delE

        return En

    def getExitData(self, state, ijList, dxList, OffSiteCount, TSOffSiteCount, beta, Nsites):

        statesTrans = np.zeros((ijList.shape[0], Nsites), dtype=int64)
        ratelist = np.zeros(ijList.shape[0])
        Specdisps = np.zeros((ijList.shape[0], self.Nspecs, 3))  # To store the displacement of each species during every jump

        siteA, specA = self.vacSiteInd, self.Nspecs - 1
        # go through all the transition

        for jumpInd in range(ijList.shape[0]):
            # Get the transition index
            siteB, specB = ijList[jumpInd], state[ijList[jumpInd]]
            transInd = self.FinSiteFinSpecJumpInd[siteB, specB]

            # copy the state
            statesTrans[jumpInd, :] = state

            # swap occupancies after jump
            statesTrans[jumpInd, siteB] = state[siteA]
            statesTrans[jumpInd, siteA] = state[siteB]

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
                OffSiteCount[interMainInd] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteB, state[siteB], interIdx]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                OffSiteCount[interMainInd] += 1

            # Next, switch required sites on
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteA, state[siteB], interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
                interMainInd = self.SiteSpecInterArray[siteB, state[siteA], interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]

            ratelist[jumpInd] = np.exp(-(0.5 * delE + delEKRA) * beta)
            Specdisps[jumpInd, specB, :] = -dxList[jumpInd, :]
            Specdisps[jumpInd, -1, :] = dxList[jumpInd, :]

            # Next, restore OffSiteCounts to original values for next jump, as well as
            # for use in the next MC sweep.
            # During switch-off operations, offsite counts were increased by one.
            # So decrease them back by one
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
                OffSiteCount[self.SiteSpecInterArray[siteA, state[siteA], interIdx]] -= 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
                OffSiteCount[self.SiteSpecInterArray[siteB, state[siteB], interIdx]] -= 1

            # During switch-on operations, offsite counts were decreased by one.
            # So increase them back by one
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
                OffSiteCount[self.SiteSpecInterArray[siteA, state[siteB], interIdx]] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
                OffSiteCount[self.SiteSpecInterArray[siteB, state[siteA], interIdx]] += 1

        return statesTrans, ratelist, Specdisps


KMC_additional_spec = [
    ("siteIndtoR", int64[:, :]),
    ("RtoSiteInd", int64[:, :, :]),
    ("N_unit", int64),
]


@jitclass(MonteCarloSamplerSpec+KMC_additional_spec)
class KMC_JIT(object):

    def __init__(self, numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts,
                 VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray,
                 numSitesTSInteracts, TSInteractSites, TSInteractSpecs, jumpFinSites, jumpFinSpec,
                 FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, JumpInteracts, Jump2KRAEng,
                 siteIndtoR, RtoSiteInd, N_unit):

        self.numSitesInteracts, self.SupSitesInteracts, self.SpecOnInteractSites, self.Interaction2En, self.numVecsInteracts, \
        self.VecsInteracts, self.VecGroupInteracts, self.numInteractsSiteSpec, self.SiteSpecInterArray = \
            numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, \
            VecsInteracts, VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray

        self.numSitesTSInteracts, self.TSInteractSites, self.TSInteractSpecs = \
            numSitesTSInteracts, TSInteractSites, TSInteractSpecs

        self.jumpFinSites, self.jumpFinSpec, self.FinSiteFinSpecJumpInd, self.numJumpPointGroups, self.numTSInteractsInPtGroups, \
        self.JumpInteracts, self.Jump2KRAEng = \
            jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, \
            JumpInteracts, Jump2KRAEng

        self.RtoSiteInd = RtoSiteInd
        self.siteIndtoR = siteIndtoR

        self.Nsites, self.Nspecs = numInteractsSiteSpec.shape[0], numInteractsSiteSpec.shape[1]

        self.N_unit = N_unit

    def TranslateState(self, state, siteFin, siteInit):
        """
        To take a state, and translate it, so the the species at siteInit
        is taken to siteFin
        :param state: The state to translate
        :param N_unit: Number of unit cells in each direction
        :return:
        """
        dR = self.siteIndtoR[siteFin, :] - self.siteIndtoR[siteInit, :]
        stateTrans = np.zeros_like(state, dtype=int64)
        for siteInd in range(state.shape[0]):
            Rnew = (self.siteIndtoR[siteInd, :] + dR) % self.N_unit  # to apply PBC
            siteIndNew = self.RtoSiteInd[Rnew[0], Rnew[1], Rnew[2]]
            stateTrans[siteIndNew] = state[siteInd]
        return stateTrans

    def GetOffSite(self, state):
        """
        :param state: State for which to count off sites of interactions
        :return: OffSiteCount array (N_interaction x 1)
        """
        OffSiteCount = np.zeros(self.numSitesInteracts.shape[0], dtype=int64)
        for interactIdx in range(self.numSitesInteracts.shape[0]):
            for intSiteind in range(self.numSitesInteracts[interactIdx]):
                if state[self.SupSitesInteracts[interactIdx, intSiteind]] !=\
                        self.SpecOnInteractSites[interactIdx, intSiteind]:
                    OffSiteCount[interactIdx] += 1
        return OffSiteCount

    def GetTSOffSite(self, state):
        """
        :param state: State for which to count off sites of TS interactions
        :return: OffSiteCount array (N_interaction x 1)
        """
        TransOffSiteCount = np.zeros(self.numSitesTSInteracts.shape[0], dtype=int64)
        for TsInteractIdx in range(self.numSitesTSInteracts.shape[0]):
            for Siteind in range(self.numSitesTSInteracts[TsInteractIdx]):
                if state[self.TSInteractSites[TsInteractIdx, Siteind]] != self.TSInteractSpecs[TsInteractIdx, Siteind]:
                    TransOffSiteCount[TsInteractIdx] += 1
        return TransOffSiteCount

    def getKRAEnergies(self, state, TSOffSiteCount, ijList):

        delEKRA = np.zeros(ijList.shape[0], dtype=float64)
        for jumpInd in range(ijList.shape[0]):
            delE = 0.0
            # Get the transition index
            siteB, specB = ijList[jumpInd], state[ijList[jumpInd]]
            transInd = self.FinSiteFinSpecJumpInd[siteB, specB]
            # We need to go through every point group for this jump
            for tsPtGpInd in range(self.numJumpPointGroups[transInd]):
                for interactInd in range(self.numTSInteractsInPtGroups[transInd, tsPtGpInd]):
                    # Check if this interaction is on
                    interactMainInd = self.JumpInteracts[transInd, tsPtGpInd, interactInd]
                    if TSOffSiteCount[interactMainInd] == 0:
                        delE += self.Jump2KRAEng[transInd, tsPtGpInd, interactInd]
            delEKRA[jumpInd] = delE

        return delEKRA

    def getEnergyChangeJumps(self, state, OffSiteCount, siteA, jmpFinSiteListTrans):

        delEArray = np.zeros(jmpFinSiteListTrans.shape[0], dtype=float64)

        for jmpInd in range(jmpFinSiteListTrans.shape[0]):
            siteB = jmpFinSiteListTrans[jmpInd]
            delE = 0.0
            # Switch required sites off
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
                # check if an interaction is on
                interMainInd = self.SiteSpecInterArray[siteA, state[siteA], interIdx]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                OffSiteCount[interMainInd] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteB, state[siteB], interIdx]
                if OffSiteCount[interMainInd] == 0:
                    delE -= self.Interaction2En[interMainInd]
                OffSiteCount[interMainInd] += 1

            # Next, switch required sites on
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
                interMainInd = self.SiteSpecInterArray[siteA, state[siteB], interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
                interMainInd = self.SiteSpecInterArray[siteB, state[siteA], interIdx]
                OffSiteCount[interMainInd] -= 1
                if OffSiteCount[interMainInd] == 0:
                    delE += self.Interaction2En[interMainInd]

            delEArray[jmpInd] = delE

            # Now revert offsitecounts for next jump
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
                OffSiteCount[self.SiteSpecInterArray[siteA, state[siteA], interIdx]] -= 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
                OffSiteCount[self.SiteSpecInterArray[siteB, state[siteB], interIdx]] -= 1

            # During switch-on operations, offsite counts were decreased by one.
            # So increase them back by one
            for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
                OffSiteCount[self.SiteSpecInterArray[siteA, state[siteB], interIdx]] += 1

            for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
                OffSiteCount[self.SiteSpecInterArray[siteB, state[siteA], interIdx]] += 1

        return delEArray

    def updateState(self, state, OffSiteCount, siteA, siteB):

        # update offsitecounts
        for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteA]]):
            interMainInd = self.SiteSpecInterArray[siteA, state[siteA], interIdx]
            OffSiteCount[interMainInd] += 1

        for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteB]]):
            interMainInd = self.SiteSpecInterArray[siteB, state[siteB], interIdx]
            OffSiteCount[interMainInd] += 1

        # Next, switch required sites on
        for interIdx in range(self.numInteractsSiteSpec[siteA, state[siteB]]):
            interMainInd = self.SiteSpecInterArray[siteA, state[siteB], interIdx]
            OffSiteCount[interMainInd] -= 1

        for interIdx in range(self.numInteractsSiteSpec[siteB, state[siteA]]):
            interMainInd = self.SiteSpecInterArray[siteB, state[siteA], interIdx]
            OffSiteCount[interMainInd] -= 1

        # swap sites
        temp = state[siteA]
        state[siteA] = state[siteB]
        state[siteB] = temp

    def getTraj(self, state, offsc, vacSiteFix, jumpFinSiteList, dxList, NSpec, Nsteps, beta):

        X = np.zeros((NSpec, 3), dtype=float64)
        t = 0.

        X_steps = np.zeros((Nsteps, NSpec, 3), dtype=float64)
        t_steps = np.zeros(Nsteps, dtype=float64)

        jumpFinSiteListTrans = np.zeros_like(jumpFinSiteList, dtype=int64)
        vacIndNow = vacSiteFix

        for step in range(Nsteps):

            # Translate the states so that vacancy is taken from vacIndnow to vacSiteFix
            stateTrans = self.TranslateState(state, vacSiteFix, vacIndNow)
            TSoffsc = self.GetTSOffSite(stateTrans)

            delEKRA = self.getKRAEnergies(stateTrans, TSoffsc, jumpFinSiteList)

            dR = self.siteIndtoR[vacIndNow] - self.siteIndtoR[vacSiteFix]

            for jmp in range(jumpFinSiteList.shape[0]):
                RfinSiteNew = (dR + self.siteIndtoR[jumpFinSiteList[jmp]]) % self.N_unit
                jumpFinSiteListTrans[jmp] = self.RtoSiteInd[RfinSiteNew[0], RfinSiteNew[1], RfinSiteNew[2]]

            delE = self.getEnergyChangeJumps(state, offsc, vacIndNow, jumpFinSiteListTrans)

            rates = np.exp(-(0.5 * delE + delEKRA) * beta)
            rateTot = np.sum(rates)
            t += 1.0/rateTot

            rates /= rateTot
            rates_cm = np.cumsum(rates)
            rn = np.random.rand()
            jmpSelect = np.searchsorted(rates_cm, rn)

            vacIndNext = jumpFinSiteListTrans[jmpSelect]

            X[NSpec - 1, :] += dxList[jmpSelect]
            specB = state[vacIndNext]
            X[specB, :] -= dxList[jmpSelect]

            X_steps[step, :, :] = X.copy()
            t_steps[step] = t

            self.updateState(state, offsc, vacIndNow, vacIndNext)

            vacIndNow = vacIndNext

        return X_steps, t_steps


# Here, we write a function that takes forms the shells
def makeShells(MC_jit, KMC_jit, state0, offsc0, TSoffsc0, ijList, dxList, beta, Nsites, Nspec, Nshells=1):
    """
    Function to make shells around a "seed" state
    :param MC_jit: JIT class MC sampler - to use the exitstates function
    :param KMC_jit: KMC Jit Class - to use state translations, offsite counters etc.
    The MC and KMC Jit classes need to be initialized with the same arrays.
    :param state0: The starting initial state.
    :param Nshells: The number of shells to build
    :return:
    """
    vacSiteInd = MC_jit.vacSiteInd

    nextShell = set([])

    state2Index = {}
    Index2State = {}
    TransitionRates = {}
    velocities = {}

    count = 0

    state0bytes = state0.tobytes()
    state2Index[state0bytes] = count
    Index2State[count] = state0

    statesTrans0, ratelist0, Specdisps0 = MC_jit.getExitData(state0, ijList, dxList, offsc0, TSoffsc0,
                                                                    beta, Nsites)

    # build the first shell
    TransitionRates[(0, 0)] = (-np.sum(ratelist0), -1)
    vel = np.zeros(3)
    for jumpInd, exitState0 in enumerate(statesTrans0):
        # assign a new index this exit state
        count += 1

        # translate it so that the vac is at the origin
        origVac = KMC_jit.TranslateState(exitState0, vacSiteInd, ijList[jumpInd])

        # Hash the state from memory buffer
        origVacbytes = origVac.tobytes()
        nextShell.add(origVacbytes)
        state2Index[origVacbytes] = count
        Index2State[count] = origVac

        # store the transition rate to this state.
        TransitionRates[(0, count)] = (ratelist0[jumpInd], jumpInd)
        vel += ratelist0[jumpInd] * dxList[jumpInd]

    TransitionsZero = TransitionRates.copy()

    velocities[state2Index[state0.tobytes()]] = vel.copy()
    lastShell = nextShell.copy()

    for shell in range(Nshells - 1):
        nextshell = set([])
        for stateBin in lastShell:
            state = np.frombuffer(stateBin, dtype=state0.dtype)
            offsc = KMC_jit.GetOffSite(state)
            TSoffsc = KMC_jit.GetTSOffSite(state)

            # Now get the exits out of this state
            statesTrans, ratelist, Specdisps = MC_jit.getExitData(state, ijList, dxList, offsc,
                                                                         TSoffsc, beta, Nsites)

            TransitionRates[(state2Index[stateBin], state2Index[stateBin])] = (-np.sum(ratelist), -1)
            vel = np.zeros(3)
            for jumpInd, exitState in enumerate(statesTrans):
                origVac = KMC_jit.TranslateState(exitState, vacSiteInd, ijList[jumpInd])
                origVacbytes = origVac.tobytes()
                if not origVacbytes in state2Index:
                    count += 1
                    state2Index[origVacbytes] = count
                    Index2State[count] = origVac

                TransitionRates[(state2Index[stateBin], state2Index[origVacbytes])] = (ratelist[jumpInd], jumpInd)
                vel += ratelist[jumpInd] * dxList[jumpInd]
                nextShell.add(origVacbytes)

            velocities[state2Index[stateBin]] = vel.copy()

        lastShell = nextShell.copy()

    return state2Index, Index2State, TransitionRates, TransitionsZero, velocities





