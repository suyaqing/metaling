# MATLAB scripts for linguistic and metalinguistic processing models. Requires spm12 for some auxiliary functions
Two processing models are included at the moment:
*ling: linguistic processing only, i.e. no contextual knowledge
*metaling1: additional contextual knowledge is included for the association between agent and location cues
The model uses reduced sentence spectrograms as input, which are stored in metaling.mat.
In addition, it has a dictionary that maps each lemma with its syllabic composition and semantic. This dictionary is stored in knowledge_MEG.mat.

The following description takes metaling1 for an example.
To simulate response to single sentences with `metaling1 (or `ling), you can either call the function `DEMP_MDP_metaling1_all(), passing the integer id of the sentence, the dictionary, and the data stored in metaling.mat using `data=load(metaling.mat). Or, you can run is as a script section by section by providing an explicit sentence id.

The main body of the DEM script defines the generative model, including its priors (D), state transition matrices (B), and likelihood (A).
It then calls the function `spm_MDP_VB_X_metaling1_all() for the variational Bayesian inference.
At last, `spm_MDP_VB_ERP_ALL_metaling() is called for plotting. It requires auxiliary functions `spm_MDP_VB_ERP_YS(), spm_MDP_VB_ERP_metaling(), spm_MDP_VB_ERP_align()
