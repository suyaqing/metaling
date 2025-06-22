function MDP = DEM_MDP_metaling2_discrete(sen)
%% syntax at the same level as semantics, and represented as transiion
% matrices. semantic+syntax directly output words/phrases
%__________________________________________________________________________
%--------------------------------------------------------------------------
rng('default')
% close all
clear
% sen = [1 16 20 5 9];
% sen = [1 17 20 6 11];
sen  = [2 5 9 19 16];
prediction = 1;
d = load('Knowledge_MEG.mat');
dict = d.dictionary_irony;
% dict = dict(1:20);
clear d;
% fname = 'speech_expand2.mat';
% 
% f = load(fname);
% % if nargin
%     input = f.sentences{sen};
% else
%     input = f.sig_fixsylb_extend{1};
% end
clear f
% alist = {'card', 'serve', 'run', 'neckband', 'score', 'buzz', ...
%     'supporters', 'cooling leaves', 'pipes', 'message', 'music key' 'token'};
wlist = {dict.Word};

% %------Level zero, syllable to spectrotemporal stripe (attracor)-----------
% D{1} = ones(numel(slist), 1); % which syllable
% D{2} = [1 0 0 0 0 0 0 0]'; % where (one of the 8 gamma period)
% % D{3} = [1 0]'; % flag for the end of the word (f3=2)
% 
% Nf = numel(D);
% for f = 1:Nf
%     Ns(f) = numel(D{f});
% end
% for f1 = 1:Ns(1)
%     for f2 = 1:Ns(2) 
% %         for f3 = 1:Ns(3)
%         % index for the I vector
% %             if f3==1
%                 A{1}((f1-1)*8+f2, f1, f2) = 1;
% %             else
% %                 A{1}(15*8+f2, f1, f2) = 1;
% %             end
% %         end
%     end
% %     A{1}((f1-1)*8+1:8, f1, 1:8) = A{1}((f1-1)*8+1:8, f1, 1:8)+1/exp(2);
% end
% 
% for f = 1:Nf
%     B{f} = eye(Ns(f));
% end
%  
% % controllable gamma location: move to the next location
% %--------------------------------------------------------------------------
% B{2}(:,:,1) = spm_speye(Ns(2),Ns(2),-1); 
% B{2}(end,end,1) = 1;
% % B{2} = B{2} + eye(Ns(2))/exp(1) + spm_speye(Ns(2),Ns(2),-2)/exp(1) ...
% %     + spm_speye(Ns(2),Ns(2),-3)/exp(2) + spm_speye(Ns(2),Ns(2),-4)/exp(3);
% % % add inprecision
% B{1} = B{1};
% 
% [DEM, demi] = spm_MDP_DEM_speech_gamma(input, fname);
% 
% % MDP Structure
% %--------------------------------------------------------------------------
% mdp.T = 8;                      % number of updates
% mdp.A = A;                      % observation model
% mdp.B = B;                      % transition probabilities
% mdp.D = D;                      % prior over initial states
% 
% mdp.Aname = {'gamma'};
% mdp.Bname = {'Sylb','gamma'};
% % mdp.chi   = 1/16;
% mdp.tau   = 4;
% mdp.DEM   = DEM;
% mdp.demi  = demi;
% mdp.label.name{1} = slist;
% 
% % for k = 1:4
% %     mdp(k) = mdp(1);
% %     mdp(k).o = [syllables(k, :); 1:4];
% % end
% 
% MDP = spm_MDP_check(mdp);
% clear mdp
% 
% clear A B D
% %------Level one, syllable to word-----------------------------------------
% D{1} = ones(numel(wlist), 1); % what word, excluding the null 
% D{2} = [1 0 0]';
% Nf = numel(D);
% for f = 1:Nf
%     Ns(f) = numel(D{f});
% end
% for f1 = 1:Ns(1)
%     s1 = dict(f1).Sylb1;
%     s2 = dict(f1).Sylb2;
%     s3 = dict(f1).Sylb3;
%     
%     idx1 = find(strcmp(slist, s1));
%     A{1}(idx1, f1, 1) = 1;
%     
%     idx2 = find(strcmp(slist, s2));
%     A{1}(idx2, f1, 2) = 1;
%     
%     idx3 = find(strcmp(slist, s3));
%     A{1}(idx3, f1, 3) = 1;
% 
% end
% Ng    = numel(A);
% for f = 1:Nf
%     B{f} = eye(Ns(f));
% end
% 
% B{1}(:, :, 1) = B{1}(:, :, 1)+0.005;
% 
% B{2}(:,:,1) = spm_speye(Ns(2),Ns(2),-1); 
% B{2}(end,end,1) = 1;
%  
%  
% % MDP Structure
% %--------------------------------------------------------------------------
% mdp.T = 3;                      % number of updates
% mdp.A = A;                      % observation model
% mdp.B = B;                      % transition probabilities
% mdp.D = D;                      % prior over initial states
% % mdp.U = U;
% % mdp.o = [syllables(1, :); 1:4];
% mdp.Aname = {'what sylb'};
% mdp.Bname = {'Lemma', 'where'};
% % mdp.chi   = 1/16;
% mdp.tau   = 4;
% mdp.MDP  = MDP;
% mdp.label.name{1} = wlist(1:end);
% mdp.link = sparse([1], [1], 1,numel(MDP(1).D),Ng);
%  
% MDP = spm_MDP_check(mdp);
% clear mdp
% 
% 
% clear A B D

% level three: association--possible combinations and their probability
%==========================================================================

% prior beliefs about initial states (in terms of counts_: D and d
%--------------------------------------------------------------------------
% top level: from irony to valence
D{1} = [1 1.01]'; % initial state for irony
M = zeros(4, 2); % mapping from irony to valence
M(1:2, 1) = 1;
M(3:4, 2) = 1;
M = spm_norm_exp(M);
D{1} = spm_norm_exp(D{1});
% discourse level: valence, context, sentence order
context{1} = 'marathon'; context{2} = 'social'; context{3} = 'money';
% context{5} = 'novel'; 

% D{1} = [1 1 1 1]';
% valence state of the discourse, determined by irony
% 1: pos event + neg verdict; 2: neg event + pos verdict
% 3: pos event + pos verdict; 4: neg event + neg verdict

D{2} = [1 1 1]'; % context, marathon, social, money
D{3} = M*D{1};
D{4} = [1 1]'; % sentence order

aglist = {'Sara', 'Tim', 'Pierre'};
aclist = {'run', 'greet', 'spend'}; % action
modlist = {'every day', 'give up', 'with smile', 'with frown', 'charity', 'gambling'}; % modifier for action 
verlist = {'persistent', 'hate running', 'friendly', 'mean', 'generous', 'unreliable'};
 
% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
Nf    = numel(D);
nag = length(aglist);
nac = length(aclist);
nm = length(modlist);
nv = length(verlist);
% nm = length(adlist);
for f = 1:Nf
    Ns(f) = numel(D{f}); 
end
for f1 = 1:Ns(2) % context
    for f2 = 1:Ns(3) % valence
%         for f3 = 1:Ns(3) % order, does not matter here
            L{1}(1:nag, f1, f2) = 1/nag; % pick any agent
            L{2}(f1, f1, f2) = 1; % context-action association
            if f2==1 % pos event + neg verdict
                L{3}(f1*2-1, f1, f2) = 1;
                L{4}(f1*2, f1, f2) = 1;        
            end
            if f2==2 % neg event + pos verdict
                L{3}(f1*2, f1, f2) = 1;
                L{4}(f1*2-1, f1, f2) = 1;      
            end
            if f2==3 % pos event + pos verdict
                L{3}(f1*2-1, f1, f2) = 1;
                L{4}(f1*2-1, f1, f2) = 1;               
            end
            if f2==4 % pos event + pos verdict
                L{3}(f1*2, f1, f2) = 1;
                L{4}(f1*2, f1, f2) = 1;               
            end
%         end
    end
            
end
% define Z
Z = cell(1, 5); % S1, V, adv, S2, adj
Z{1}(:, 1) = [1 0 0 0 0]'; Z{1}(:, 2) = [1 0 0 0 0]';
Z{2}(:, 1) = [0 1 0 0 0]'; Z{2}(:, 2) = [0 0 0 1 0]';
Z{3}(:, 1) = [0 0 1 0 0]'; Z{3}(:, 2) = [0 0 0 0 1]';
Z{4}(:, 1) = [0 0 0 0 1]'; Z{4}(:, 2) = [0 1 0 0 0]';
Z{5}(:, 1) = [0 0 0 1 0]'; Z{5}(:, 2) = [0 0 1 0 0]';
% calculate and normalize initial distributions for the top level
% D{1} = spm_norm_exp(D{1}, 1);
D{2} = spm_norm_exp(D{2}, 1); % context
D{3} = spm_norm_exp(D{3}, 1); % valence
D{4} = spm_norm_exp(D{4}, 1); % order
for ks = 1:4
    D{ks+4} = zeros(size(L{ks}, 1), 1);
    L{ks} = spm_norm_exp(L{ks});
    if prediction
        for kv = 1:2
            ll = squeeze(L{ks}(:, :, kv));
            D{ks+4} = D{ks+4} + D{3}(kv)*ll*D{2};
        end
    else
        D{ks+4} = ones(length(D{ks+4}), 1);
    end
    D{ks+4} = spm_norm_exp(D{ks+4});
end

D{9} = zeros(5, 5);
for tau = 1:5
    Z{tau} = spm_norm_exp(Z{tau});
    D{9}(:, tau) = spm_norm_exp(Z{tau}*D{4});
end
    
Nw = length(dict);
A = cell(1, 5);
A{1} = zeros(Nw, nag);
A{2} = zeros(Nw, nac);
A{3} = zeros(Nw, nm);
A{4} = zeros(Nw, nv); 
A{5} = zeros(Nw, nag); % "she" or "he" for the subject in sentence 2
m1 = {dict(:).Meaning1};
% m2 = {dict(:).Meaning2};
% m3 = {dict(:).Meaning3};
for f = 1:Nf
    Ns(f) = numel(D{f}); 
end

for ns = 1:nag
    sem = aglist{ns};
    idx_a = find(strcmp(m1, sem));
    A{1}(idx_a, ns) = 1;
    if ns==1 % Sara
        A{5}(Nw, ns) = 1; % "she"
    else
        A{5}(Nw-1, ns) = 1; % "he"
    end
end

for ns = 1:nac
    sem = aclist{ns};
    idx_v = find(strcmp(m1, sem));
    A{2}(idx_v, ns) = 1;
end

for ns = 1:nm
    sem = modlist{ns};
    idx_mod = find(strcmp(m1, sem));
    A{3}(idx_mod, ns) = 1;
end

for ns = 1:nv
    sem = verlist{ns};
    idx_ver = find(strcmp(m1, sem));
    A{4}(idx_ver, ns) = 1;
end

% for ns = 1:nm
%     sem = adlist{ns};
%     idx_ad = find(strcmp(m1, sem));
%     if ~isempty(idx_ad)
%         A{5}(idx_ad, ns) = 1;
%     end
% end

for ksyn = 1:5
    A{ksyn} = spm_norm_exp(A{ksyn}, 2);
end
% MDP Structure
%--------------------------------------------------------------------------
% mdp.MDP  = MDP;
% mdp.link = sparse([1 2 3 4 5],[1 2 3 4 5],[1 1 1 1 1], numel(MDP.D),Ng); % link function is a Ng1*Ng2 matrix with the (1,1) entry equals to 1
% because the first factors of both levels are linked (sentence and word)
 
mdp.T = 5;                      % number of moves
mdp.M = M;
mdp.A = A;                      % observation model
mdp.L = L;
mdp.Z = Z;
mdp.D = D;                      % prior over initial states
% the input can be defined as indexes of outcome list
mdp.o = sen;

mdp.stepr = 256;
mdp.stepc = 256; % should not be smaller than 128
mdp.stepv = 512;
mdp.stepo = 256;
mdp.stepsyn = 32;
mdp.steps = [16 16 64 64]; %[ag ac m v]
% mdp.stepa = 8;
% mdp.stepr = 16;
% mdp.stepp = 8;
% mdp.stepm = 8;

mdp.label.name{1} = {'Ironic', 'Honest'};
mdp.label.name{2} = context;
mdp.label.name{3} = {'Ironic +', 'Ironic -', 'Honest +', 'Honest -'};
mdp.label.name{4} = {'Order A', 'Order B'};
% mdp.label.name{2} = {'1', '2', '3', '4', '5', '6', '7', '8'};
mdp.label.name{5}   = aglist;
mdp.label.name{6}   = aclist;
mdp.label.name{7}   = modlist;
mdp.label.name{8}   = verlist;
% mdp.label.name{6}   = adlist;
mdp.label.name{9} = {'Subject', 'Verb', 'Adv', 'Adj', 'Pronoun'};
mdp.label.factor   = {'Irony', 'Context', 'Valence', 'Order', 'Agent', 'Action', 'Modifier', ...
    'Verdict', 'Syntax'};
% mdp         = spm_MDP_check(mdp);
%%
% illustrate a single trial
%==========================================================================
% prediction = 1;

MDP  = spm_MDP_VB_X_metaling2_discrete(mdp, prediction);
% if nargin
%     return;
% end

%% plot outcome
sentence = [dict(sen(1)).Word ' ' dict(sen(2)).Word ' ' dict(sen(3)).Word ' ' dict(sen(4)).Word ...
    ' ' dict(sen(5)).Word '.'];
spm_MDP_VB_ERP_metaling2_discrete(MDP, sentence);
% plot_info_passing(MDP, 'IT')
 

% spm_MDP_VB_ERP_ALL_hybrid(MDP)

% figure;
% spm_MDP_VB_ERP_YS(MDP.mdp(4).mdp, 2)

% spm_figure('GetWin','Figure 2'); clf
% spm_MDP_VB_LFP(MDP.mdp(4).mdp.mdp,[], 1); 
% 
