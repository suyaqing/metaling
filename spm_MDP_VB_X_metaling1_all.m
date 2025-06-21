function [MDP] = spm_MDP_VB_X_metaling1_all(MDP, prediction)
% active inference and learning using variational message passing
% FORMAT [MDP] = spm_MDP_VB_X(MDP,OPTIONS)

% set up and preliminaries
%==========================================================================
global ichunk
% defaults
%--------------------------------------------------------------------------
try, stepo   = MDP.stept;   catch, stepo = 256;    end % learning rate
try, stepc   = MDP.stepc;   catch, stepc   = 256;    end % update time constant
try, stepsyn   = MDP.stepsyn;   catch, stepsyn   = 8;    end % update time constant
try, steps   = MDP.steps;   catch, steps   = 16*ones(1,3); end % Occam window updates
try, erp   = MDP.erp;   catch, erp   = 2;    end % update reset


T = MDP.T;
D = MDP.D;
L = MDP.L;
A = MDP.A;
Z = MDP.Z;
Nc = length(D{1});
No = length(D{2});
Nw = size(A{1}, 1);

W = cell(1, T);
W0 = cell(1, T);
% initialise model-specific variables
%==========================================================================
Ni    = 16;                                % number of VB iterations

% initialize x, X, nx
Xc = zeros(Nc, T);
Xc(:, 1) = D{1};
xc = D{1};
% xc = zeros(Nc, 1);
nxc = zeros(Ni, Nc, T);
buxc = zeros(Ni, Nc, T);

Xo = zeros(No, T);
Xo(:, 1) = D{2};
xo = D{2};
% xt = zeros(Nt, 1);
nxo = zeros(Ni, No, T);
buxo = zeros(Ni, No, T);

Xs = cell(1, 3); xs = Xs; nxs = Xs; buxs = Xs; tdxs = Xs;
for ks = 1:3
    Ns = size(A{ks+1}, 2);
    Xs{ks} = zeros(Ns, T);
    Xs{ks}(:, 1) = D{ks+2};
    xs{ks} = D{ks+2};
%     xs{ks} = zeros(Ns, 1);
    nxs{ks} = zeros(Ni, Ns, T);
    buxs{ks} = zeros(Ni, Ns, T);
    tdxs{ks} = zeros(Ni, Ns, T);
end

Nsyn = size(D{end}, 1);
Xsyn = cell(1, T); xsyn = Xsyn; nxsyn = Xsyn;
buxsyn = zeros(Ni, Nsyn, T); tdxsyn = buxsyn;
for tau = 1:T
    
    Xsyn{tau} = zeros(Nsyn, 1);
%     Xsyn{tau}(:, 1) = D{end}(:, tau);
    xsyn{tau} = D{end}(:, tau);
    nxsyn{tau} = zeros(Ni, Nsyn);
end

% xc(:) = 1/length(xc);
% xo(:) = 1/length(xo);
% for ks = 1:3
%     xs{ks}(:) = 1/length(xs{ks});
% end
% for tau = 1:4
%     xsyn{tau}(:) = 1/length(xsyn{tau});
% end
F = zeros(Ni, T);

% ensure any outcome generating agent is updated first
%--------------------------------------------------------------------------
% [M,MDP] = spm_MDP_get_M(MDP,T,Ng);


% belief updating over successive time points
%==========================================================================
for t = 1:T
    
    % calculate word prediction
    %======================================================================
%     if t==1
%         W0{1} = A{1};
%     else
        W0{t} = zeros(Nw, 1);
        syn = Z{t}*Xo(:, t);
        W0{t} = W0{t}+A{1}*syn(1);
        for ksyn = 2:4
            W0{t} = W0{t}+syn(ksyn)*A{ksyn}*Xs{ksyn-1}(:, t);
        end
%     end
    try
        mdp = MDP.mdp(t);
    catch
        try
            mdp     = spm_MDP_update(MDP.MDP(t),MDP.mdp(t - 1));
        catch
            try
                mdp = spm_MDP_update(MDP.MDP(1),MDP.mdp(t - 1));
            catch
                mdp = MDP.MDP(1);
            end
        end
    end
    mdp.factor = 1;
    if prediction
        mdp.D{1} = spm_softmax(spm_log(W0{t})/erp);
        MDP.mdp(t) = spm_MDP_VB_X(mdp);
    else
%         mdp.D{1} = spm_softmax(spm_log(W0{t})/erp);
        mdp.D{1} = ones(Nw, 1)/Nw;
        OPTIONS.pred = 0;
        MDP.mdp(t) = spm_MDP_VB_X(mdp, OPTIONS);
    end

%     wo = zeros(size(W0{t}));
%     wo(MDP.o(t)) = 1; % direct lemma input
%     W{t} = wo;
    W{t} = MDP.mdp(t).X{1}(:, 1);
               
    for i = 1:Ni
        
        
        % Variational updates (skip to t = T in HMM mode)
        %==================================================================
            
            % processing time and reset
            %--------------------------------------------------------------
            tstart = tic;
           
            
            % Variational updates (hidden states) under sequential policies
            %==============================================================
            % context
            v0 = spm_log(xc);
            BU = zeros(length(v0), 1);
            for ks = 1:3
                for kso = 1:2
                    ll = xo(kso)*squeeze(spm_log(L{ks}(:, :, kso)));
                    BU = BU + ll'*xs{ks};
                end
            end
            dFdx = v0 - BU - spm_log(D{1});
%             dFdx = dFdx - mean(dFdx);
            sxc = spm_softmax(v0 - dFdx/stepc);
            F(i, t) = F(i, t) + sxc'*(spm_log(sxc) - spm_log(D{1}));
            buxc(i, :, t) = BU;
%             buxc(i, t) = sum(BU);
            
            % order
            v0 = spm_log(xo);
            BU1 = zeros(length(v0), 1);
%             BU2 = BU1;
            for tau = 1:t
                BU1 = BU1 + spm_log(Z{tau}')*xsyn{tau};
            end
%             for ks = 1:3
%                 for kso = 1:2
%                     ll = xs{ks}'*squeeze(spm_log(L{ks}(:, :, kso)));
%                     BU2(kso) = BU2(kso) + ll*xc;
%                 end
%             end
            dFdx = v0 - BU1 - spm_log(D{2});
%             dFdx = dFdx - mean(dFdx);
            so = spm_softmax(v0 - dFdx/stepo);
            F(i, t) = F(i, t) + so'*(spm_log(so) - spm_log(D{2}));
            buxo(i, :, t) = BU1;
%             buxo(i, t) = sum(BU1+BU2);
            
            % syntax
            for tau = 1:t
                v0 = spm_log(xsyn{tau});
                BU = zeros(length(v0), 1);
                BU(1) = W{tau}'*spm_log(A{1});
                for ksyn = 2:4
                    BU(ksyn) = W{tau}'*(spm_log(A{ksyn})*xs{ksyn-1});
                end
                TD = spm_log(Z{tau})*xo;
                dFdx = v0 - BU - TD;
%                 dFdx = dFdx - mean(dFdx);
                sxsyn{tau} = spm_softmax(v0 - dFdx/stepsyn);
                F(i, t) = F(i, t) + sxsyn{tau}'*(spm_log(sxsyn{tau}) - spm_log(Z{tau})*so);
                buxsyn(i, :, t) = BU;
                tdxsyn(i, :, t) = TD;
%                 buxsyn(i, tau) = sum(BU);
%                 tdxsyn(i, tau) = sum(TD);
            end
            
            % semantic
            for ks = 1:3
                v0 = spm_log(xs{ks});
                BU = zeros(length(v0), 1);
                TD = BU;
                ww = zeros(Nw, 1);
                if t>=1
                    for tau = 1:t
                        ww = ww + W{tau}*xsyn{tau}(ks+1);
                    end
                end
                BU = spm_log(A{ks+1}')*ww;
                for kso = 1:2
                    ll = xo(kso)*squeeze(spm_log(L{ks}(:, :, kso)));
                    TD = TD + ll*xc;
                end
                dFdx = v0 - BU - TD;
%                 dFdx = dFdx - mean(dFdx);
                sxs{ks} = spm_softmax(v0 - dFdx/steps(ks));
                F(i, t) = F(i, t) + sxs{ks}'*spm_log(sxs{ks});
                for kso = 1:2
                    ll = so(kso)*squeeze(spm_log(L{ks}(:, :, kso)));
                    F(i, t) = F(i, t) - sxs{ks}'*(ll*sxc);
                end
                buxs{ks}(i, :, t) = BU;
                tdxs{ks}(i, :, t) = TD;
%                 buxs{ks}(i, t) = sum(BU);
%                 tdxs{ks}(i, t) = sum(TD);
            end     
            
            
            xc = sxc;
            Xc(:, t) = sxc;
            nxc(i, :, t) = sxc;
            if t<T
                Xc(:, t+1) = sxc;
            end
            
            xo = so;
            Xo(:, t) = so;
            nxo(i, :, t) = so;
            if t<T
                Xo(:, t+1) = so;
            end
            
            for ks = 1:3
                xs{ks} = sxs{ks};
                Xs{ks}(:, t) = sxs{ks};
                nxs{ks}(i, :, t) = sxs{ks};
                if t<T
                    Xs{ks}(:, t+1) = Xs{ks}(:, t);
                end
            end
            
            for tau = 1:t
                xsyn{tau} = sxsyn{tau};
                Xsyn{tau}(:) = sxsyn{tau};
                nxsyn{tau}(i, :) = sxsyn{tau};
            end
            % Free energy
            %--------------------------------------------------------------
            for tau = 1:t
                F(i, t) = F(i, t) - xsyn{tau}(1)*W{tau}'*spm_log(A{1});
                for ksyn = 2:4 % need to check, first one not always attribute
                    F(i, t) = F(i, t) - xsyn{tau}(ksyn)*W{tau}'*(spm_log(A{ksyn})*xs{ksyn-1});
                end
            end
           
    end
    
%     xc(:) = 1/length(xc);
%     xo(:) = 1/length(xo);
%     for ks = 1:3
%         xs{ks}(:) = 1/length(xs{ks});
%     end
%     for tau = 1:4
%         xsyn{tau}(:) = 1/length(xsyn{tau});
%     end
   
            
            
            
    % processing (i.e., reaction) time
    %--------------------------------------------------------------
    rt(t)      = toc(tstart);


end % end of loop over time


    
    % assemble results and place in NDP structure
    %----------------------------------------------------------------------
   
MDP.Xc  = Xc;       % Bayesian model averages over T outcomes
MDP.nxc = nxc;
MDP.buxc = buxc;
MDP.Xt = Xo;
MDP.nxo = nxo;
MDP.buxo = buxo;
MDP.Xs = Xs;
MDP.nxs = nxs;
MDP.buxs = buxs;
MDP.tdxs = tdxs;
MDP.Xsyn = Xsyn;
MDP.nxsyn = nxsyn;
MDP.buxsyn = buxsyn;
MDP.tdxsyn = tdxsyn;
MDP.W = W;
MDP.W0 = W0;
MDP.F = F;

MDP.rt = rt;        % simulated reaction time (seconds)
    



% auxillary functions
%==========================================================================

function A  = spm_log(A)
% log of numeric array plus a small constant
%--------------------------------------------------------------------------
A  = log(A + 1e-16);

function A  = spm_norm(A, mode)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A           = bsxfun(@rdivide,A,sum(A,1));
if nargin<2 || mode==1
    A(isnan(A)) = 1/size(A,1);
else
    A(isnan(A)) = 0;
end

function A  = spm_wnorm(A)
% summation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A   = A + 1e-16;
A   = bsxfun(@minus,1./sum(A,1),1./A)/2;

function sub = spm_ind2sub(siz,ndx)
% subscripts from linear index
%--------------------------------------------------------------------------
n = numel(siz);
k = [1 cumprod(siz(1:end-1))];
for i = n:-1:1,
    vi       = rem(ndx - 1,k(i)) + 1;
    vj       = (ndx - vi)/k(i) + 1;
    sub(i,1) = vj;
    ndx      = vi;
end

return



function [M,MDP] = spm_MDP_get_M(MDP,T,Ng)
% FORMAT [M,MDP] = spm_MDP_get_M(MDP,T,Ng)
% returns an update matrix for multiple models
% MDP(m) - structure array of m MPDs
% T      - number of trials or updates
% Ng(m)  - number of output modalities for m-th MDP
%
% M      - update matrix for multiple models
% MDP(m) - structure array of m MPDs
%
% In some applications, the outcomes are generated by a particular model
% (to maximise free energy, based upon the posterior predictive density).
% The generating model is specified in the matrix MDP(m).n, with a row for
% each outcome modality, such that each row lists the index of the model
% responsible for generating outcomes.
%__________________________________________________________________________

% check for VOX and ensure the agent generates outcomes when speaking
%--------------------------------------------------------------------------
if numel(MDP) == 1
    if isfield(MDP,'MDP')
        if isfield(MDP.MDP,'VOX')
            MDP.n = [MDP.MDP.VOX] == 1;
        end
    end
end
    
for m = 1:size(MDP,1)
    
    % check size of outcome generating agent, as specified by MDP(m).n
    %----------------------------------------------------------------------
    if ~isfield(MDP(m),'n')
        MDP(m).n = zeros(Ng(m),T);
    end
    if size(MDP(m).n,1) < Ng(m)
        MDP(m).n = repmat(MDP(m).n(1,:),Ng(m),1);
    end
    if size(MDP(m).n,1) < T
        MDP(m).n = repmat(MDP(m).n(:,1),1,T);
    end
    
    % mode of generating model (most frequent over outcome modalities)
    %----------------------------------------------------------------------
    n(m,:) = mode(MDP(m).n.*(MDP(m).n > 0),1);
    
end

% reorder list of model indices for each update
%--------------------------------------------------------------------------
n     = mode(n,1);
for t = 1:T
    if n(t) > 0
        M(t,:) = circshift((1:size(MDP,1)),[0 (1 - n(t))]);
    else
        M(t,:) = 1;
    end
end


return

function MDP = spm_MDP_update(MDP,OUT)
% FORMAT MDP = spm_MDP_update(MDP,OUT)
% moves Dirichlet parameters from OUT to MDP
% MDP - structure array (new)
% OUT - structure array (old)
%__________________________________________________________________________

% check for concentration parameters at this level
%--------------------------------------------------------------------------
try,  MDP.a = OUT.a; end
try,  MDP.b = OUT.b; end
try,  MDP.c = OUT.c; end
try,  MDP.d = OUT.d; end
try,  MDP.e = OUT.e; end

% check for concentration parameters at nested levels
%--------------------------------------------------------------------------
try,  MDP.MDP(1).a = OUT.mdp(end).a; end
try,  MDP.MDP(1).b = OUT.mdp(end).b; end
try,  MDP.MDP(1).c = OUT.mdp(end).c; end
try,  MDP.MDP(1).d = OUT.mdp(end).d; end
try,  MDP.MDP(1).e = OUT.mdp(end).e; end

return




