function MDP = DEM_MDP_ling_all(senid, dict, f)
%% syntax at the same level as semantics, and represented as transiion
% matrices. semantic+syntax directly output words/phrases
%__________________________________________________________________________
%--------------------------------------------------------------------------
rng('default')
% close all
%clear
global ichunk
ichunk = 0;
prediction = 1;
fname = 'speech_metaling.mat'; 
if nargin<=1
    if ~nargin 
        senid = 5;
    end
    d = load('Knowledge_MEG.mat');
    dict = d.dictionary; clear d
    f = load(fname);
    input = f.sentences_clean{senid};
    sen = f.sen{senid};
else 
    input = f.sentences_clean{senid};
    sen = f.sen{senid};
    clear f
end
% % if nargin
%     input = f.sentences{sen};
% else
%     input = f.sig_fixsylb_extend{1};
% end
% clear f

alist = {'Sara', 'Tim', 'Pierre'};
vlist = {'talk', 'help', 'eat'};
lolist = {'playground','classroom', 'auditorium', 'lab', 'counter', 'shop', 'cashier', 'warehouse', ...
    'custom', 'terminal', 'security', 'gate'}; % no need to have null, all are used
slist = {'ai', 'at', 'au', 'be', 'boi', 'bor', 'di', 'ding', 'e', 'geit', 'grund', 'haind', 'haus', ...
    'helpt', 'hi', 'in', 'ji', 'ka', 'kawn', 'king', 'kju', 'klas', 'kus', 'lab', 'mi', 'nal', ...
    'nir', 'no', 'pjir', 'plei', 'ra', 'ri', 'rium', 'rum', 'sa', 'sau', 'se', 'shir', 'shop', ...
    'tal', 'tem', 'ter', 'ti', 'tim', 'ting', 'tist', 'tor', 'ware', 'wez', 'zat', 'ze', '-'};

wlist = {dict.Word};

% %------Level zero, syllable to spectrotemporal stripe (attracor)-----------
D{1} = ones(numel(slist), 1); % which syllable
D{2} = [1 0 0 0 0 0 0 0]'; % where (one of the 8 gamma period)
% D{3} = [1 0]'; % flag for the end of the word (f3=2)

Nf = numel(D);
for f = 1:Nf
    Ns(f) = numel(D{f});
end
for f1 = 1:Ns(1)
    for f2 = 1:Ns(2) 
        A{1}((f1-1)*8+f2, f1, f2) = 1;          
    end
end

for f = 1:Nf
    B{f} = eye(Ns(f));
end
 
% controllable gamma location: move to the next location
%--------------------------------------------------------------------------
B{2}(:,:,1) = spm_speye(Ns(2),Ns(2),-1); 
B{2}(end,end,1) = 1;
% B{2} = B{2} + eye(Ns(2))/exp(1) + spm_speye(Ns(2),Ns(2),-2)/exp(1) ...
%     + spm_speye(Ns(2),Ns(2),-3)/exp(2) + spm_speye(Ns(2),Ns(2),-4)/exp(3);
% % add inprecision
B{1} = B{1};

[DEM, demi] = spm_MDP_DEM_speech_gamma(input, fname);
% 
% % MDP Structure
% %--------------------------------------------------------------------------
mdp.T = 8;                      % number of updates
mdp.A = A;                      % observation model
mdp.B = B;                      % transition probabilities
mdp.D = D;                      % prior over initial states

mdp.Aname = {'gamma'};
mdp.Bname = {'Sylb','gamma'};
% mdp.chi   = 1/16;
mdp.tau   = 4;
mdp.DEM   = DEM;
mdp.demi  = demi;
mdp.label.name{1} = slist;

% for k = 1:4
%     mdp(k) = mdp(1);
%     mdp(k).o = [syllables(k, :); 1:4];
% end

MDP = spm_MDP_check(mdp);
clear mdp 
clear A B D
%------Level one, syllable to word-----------------------------------------
D{1} = ones(numel(wlist), 1); % what word, excluding the null 
D{2} = [1 0 0 0 0 0]'; % maximum of six syllables
Nf = numel(D);
for f = 1:Nf
    Ns(f) = numel(D{f});
end
for f1 = 1:Ns(1)
    s1 = dict(f1).Sylb1;
    s2 = dict(f1).Sylb2;
    s3 = dict(f1).Sylb3;
    s4 = dict(f1).Sylb4;
    s5 = dict(f1).Sylb5;
    s6 = dict(f1).Sylb6;
    
    idx1 = find(strcmp(slist, s1));
    A{1}(idx1, f1, 1) = 1;
    
    idx2 = find(strcmp(slist, s2));
    A{1}(idx2, f1, 2) = 1;
    
    idx3 = find(strcmp(slist, s3));
    A{1}(idx3, f1, 3) = 1;

    idx4 = find(strcmp(slist, s4));
    A{1}(idx4, f1, 4) = 1;
    
    idx5 = find(strcmp(slist, s5));
    A{1}(idx5, f1, 5) = 1;
    
    idx6 = find(strcmp(slist, s6));
    A{1}(idx6, f1, 6) = 1;

end
Ng    = numel(A);
for f = 1:Nf
    B{f} = eye(Ns(f));
end

B{1}(:, :, 1) = B{1}(:, :, 1)+0.005;

B{2}(:,:,1) = spm_speye(Ns(2),Ns(2),-1); 
B{2}(end,end,1) = 1;
 
 
% MDP Structure
%--------------------------------------------------------------------------
mdp.T = 6;                      % number of updates
mdp.A = A;                      % observation model
mdp.B = B;                      % transition probabilities
mdp.D = D;                      % prior over initial states
% mdp.U = U;
% mdp.o = input;
mdp.Aname = {'what sylb'};
mdp.Bname = {'Lemma', 'where'};
% mdp.chi   = 1/16;
mdp.tau   = 4;
mdp.MDP  = MDP;
mdp.label.name{1} = wlist(1:end);
mdp.link = sparse([1], [1], 1, numel(MDP(1).D),Ng); 
 
MDP = spm_MDP_check(mdp);
clear mdp


clear A B D

% level three: association--possible combinations and their probability
%==========================================================================
 
% prior beliefs about initial states (in terms of counts_: D and d
%--------------------------------------------------------------------------


D{1} = [1 1]'; % order

% alist = {'Sara', 'Tim', 'Pierre'};
% vlist = {'talk', 'help', 'eat'};
% lolist = {'playground','classroom', 'auditorium', 'lab', 'counter', 'shop', 'cashier', 'warehouse', ...
%     'custom', 'terminal', 'security', 'gate'}; 
 
% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
Nf    = numel(D);
na = length(alist);
nv = length(vlist);
nl = length(lolist);
% nm = length(adlist);
for f = 1:Nf
    Ns(f) = numel(D{f}); 
end
for f1 = 1:Ns(1) % context    
        L{1}(1:3, f1) = 1/3;
        L{2}(1:3, f1) = 1/3;
        L{3}(1:12, f1) = 1/12;
end
% define Z
Z = cell(1, 4);
Z{1}(:, 1) = [1 0 0 0]'; Z{1}(:, 2) = [0 0 0 1]';
Z{2}(:, 1) = [0 1 0 0]'; Z{2}(:, 2) = [1 0 0 0]';
Z{3}(:, 1) = [0 0 1 0]'; Z{3}(:, 2) = [0 1 0 0]';
Z{4}(:, 1) = [0 0 0 1]'; Z{4}(:, 2) = [0 0 1 0]';
% calculate and normalize initial distributions for the top level
D{1} = spm_norm_exp(D{1}, 1);
% D{2} = spm_norm_exp(D{2}, 1);
for ks = 1:3
    D{ks+1} = zeros(size(L{ks}, 1), 1);
    L{ks} = spm_norm_exp(L{ks});
    if prediction
        for kt = 1:2
            ll = squeeze(L{ks}(:, kt));
            D{ks+1} = D{ks+1} + D{1}(kt)*ll;
        end
    else
        D{ks+1} = ones(length(D{ks+1}), 1);
    end
    D{ks+1} = spm_norm_exp(D{ks+1});
end

D{5} = zeros(4, 4);
for tau = 1:4
    Z{tau} = spm_norm_exp(Z{tau});
    D{5}(:, tau) = spm_norm_exp(Z{tau}*D{1});
end
    
Nw = length(dict);
A = cell(1, 4);
A{1} = zeros(Nw, 1); A{1}(1:2) = 1; % filler
A{2} = zeros(Nw, na);
A{3} = zeros(Nw, nv);
A{4} = zeros(Nw, nl);
% A{5} = zeros(Nw, nm);
m1 = {dict(:).Meaning1};
% m2 = {dict(:).Meaning2};
% m3 = {dict(:).Meaning3};
for f = 1:Nf
    Ns(f) = numel(D{f}); % number of total possible states under each factor--YS
end

for ns = 1:na
    sem = alist{ns};
    idx_a = find(strcmp(m1, sem));
    A{2}(idx_a, ns) = 1;
end

for ns = 1:nv
    sem = vlist{ns};
    idx_v = find(strcmp(m1, sem));
    A{3}(idx_v, ns) = 1;
end

for ns = 1:nl
    sem = lolist{ns};
    idx_p1 = find(strcmp(m1, sem));
    A{4}(idx_p1, ns) = 1;
end

% for ns = 1:nm
%     sem = adlist{ns};
%     idx_ad = find(strcmp(m1, sem));
%     if ~isempty(idx_ad)
%         A{5}(idx_ad, ns) = 1;
%     end
% end

for ksyn = 1:4
    A{ksyn} = spm_norm_exp(A{ksyn}, 2);
end
% MDP Structure
%--------------------------------------------------------------------------
mdp.MDP  = MDP;
% mdp.link = sparse([1 2 3 4 5],[1 2 3 4 5],[1 1 1 1 1], numel(MDP.D),Ng); % link function is a Ng1*Ng2 matrix with the (1,1) entry equals to 1
% because the first factors of both levels are linked (sentence and word)
 
mdp.T = 4;                      % number of moves

mdp.A = A;                      % observation model
mdp.L = L;
mdp.Z = Z;
mdp.D = D;                      % prior over initial states
% the input can be defined as indexes of outcome list
% mdp.o = sen;
%%
% mdp.stepc = 256; % should not be smaller than 128
mdp.stepo = 128;
mdp.stepsyn = 8;
mdp.steps = [16 16 16]; %[a v l]
% mdp.stepa = 8;
% mdp.stepr = 16;
% mdp.stepp = 8;
% mdp.stepm = 8;

% mdp.label.name{1} = context;
mdp.label.name{1} = {'Order A', 'Order B'};
% mdp.label.name{2} = {'1', '2', '3', '4', '5', '6', '7', '8'};
mdp.label.name{2}   = alist;
mdp.label.name{3}   = vlist;
mdp.label.name{4}   = lolist;
% mdp.label.name{6}   = adlist;
mdp.label.name{5} = {'Attribute', 'Subject', 'Verb', 'Place'};
mdp.label.factor   = {'Order', 'Agent', 'Action', 'Location', 'Syntax'};
% mdp         = spm_MDP_check(mdp);
%%
% illustrate a single trial
%==========================================================================
% prediction = 1;

MDP  = spm_MDP_VB_X_ling_all(mdp, prediction);
if nargin
    return;
end

%% plot outcome
sentence = [sen{1} ' ' sen{2} ' ' sen{3} ' ' sen{4} '.'];
spm_MDP_VB_ERP_ALL_ling(MDP, sentence);
% plot_info_passing(MDP, 'IT')
 

% spm_MDP_VB_ERP_ALL_hybrid(MDP)

% figure;
% spm_MDP_VB_ERP_YS(MDP.mdp(4).mdp, 2)

% spm_figure('GetWin','Figure 2'); clf
% spm_MDP_VB_LFP(MDP.mdp(4).mdp.mdp,[], 1); 
% 
