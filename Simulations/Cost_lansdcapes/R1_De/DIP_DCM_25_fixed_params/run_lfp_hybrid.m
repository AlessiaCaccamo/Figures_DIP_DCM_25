% PhD Mathematics and Statistics, Thesis Chapter 1, Alessia Caccamo, University of Exeter, January 2024
function DCM=run_lfp_hybrid(MOGA_params,nsim,data,freq) %Return DCM structure

DCM = [];
DCM.A = {[1],[1],[1]};
DCM.B = {};
DCM.C = 0;


DCM.xY.y{1} = data;% data
DCM.xY.Hz = freq; % frequency from 1 to 20 Hz
DCM.xY.dt = 1;


options.Nmodes       =1;% number of spatial modes
%options.Tdcm         [- [start end] time window in ms

options.Fdcm         = DCM.xY.Hz([1 end]);%[start end] Frequency window in Hz
options.D            =1;%- time bin decimation       (usually 1 or 2)
options.spatial      ='LFP';%- 'ECD', 'LFP' or 'IMG'     (see spm_erp_L)
options.model        ='LFP';%- 'ERP', 'SEP', 'CMC', 'LFP', 'NMM' or 'MFM'

DCM.options = options;
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
% check options
%==========================================================================
drawnow
clear spm_erp_L
name = sprintf('DCM_%s',date);
DCM.options.analysis  = 'CSD';
 
% Filename and options
%--------------------------------------------------------------------------
try, DCM.name;                      catch, DCM.name = name;      end
try, model   = DCM.options.model;   catch, model    = 'NMM';     end
try, spatial = DCM.options.spatial; catch, spatial  = 'LFP';     end
try, Nm      = DCM.options.Nmodes;  catch, Nm       = 8;         end
try, DATA    = DCM.options.DATA;    catch, DATA     = 1;         end

display(model); 
% Spatial model
%==========================================================================
DCM.options.Nmodes = Nm;
DCM.M.dipfit.model = model;
DCM.M.dipfit.type  = spatial;

if DATA
    %DCM  = spm_dcm_erp_data(DCM);                   % data
    %DCM  = spm_dcm_erp_dipfit(DCM, 1);              % spatial model
end

     % DCM.M.dipfit % the following fields are added to the structure,
     % e.g., DCM.M.dipfit.location

   dipfit.location =0;%- 0 or 1 for source location priors
   dipfit.symmetry =0;%- 0 or 1 for symmetry constraints on sources
   dipfit.modality='LFP';%- 'EEG', 'MEG', 'MEGPLANAR' or 'LFP'
   dipfit.type    ='LFP';%- 'ECD', 'LFP' or 'IMG''
   dipfit.symm    =0;%- distance (mm) for symmetry constraints (ECD)
   dipfit.Lpos    =0;%- x,y,z source positions (mm)            (ECD)
   dipfit.Nm      =1;%- number of modes                        (Imaging)
   dipfit.Ns      =1;%- number of sources
   dipfit.Nc      =1;%- number of channels

   dipfit.model=options.model;
DCM.M.dipfit = dipfit;

Ns   = length(DCM.A{1});                            % number of sources


% Design model and exogenous inputs
%==========================================================================
if ~isfield(DCM,'xU'),   DCM.xU.X = sparse(1 ,0); end
if ~isfield(DCM.xU,'X'), DCM.xU.X = sparse(1 ,0); end
if ~isfield(DCM,'C'),    DCM.C    = sparse(Ns,0); end
if isempty(DCM.xU.X),    DCM.xU.X = sparse(1 ,0); end
if isempty(DCM.xU.X),    DCM.C    = sparse(Ns,0); end

% Neural mass model
%==========================================================================
 
% prior moments on parameters
%--------------------------------------------------------------------------
[pE,pC]  = spm_dcm_neural_priors(DCM.A,DCM.B,DCM.C,model);
  
% check to see if neuronal priors have already been specified
%--------------------------------------------------------------------------
try
    if spm_length(DCM.M.pE) == spm_length(pE);
        pE = DCM.M.pE;
        pC = DCM.M.pC;
        fprintf('Using existing priors\n')
    end
end

% augment with priors on spatial model
%--------------------------------------------------------------------------
[pE,pC] = spm_L_priors(DCM.M.dipfit,pE,pC);
 
% augment with priors on endogenous inputs (neuronal) and noise
%--------------------------------------------------------------------------
[pE,pC] = spm_ssr_priors(pE,pC);

try
    if spm_length(DCM.M.pE) == spm_length(pE);
        pE = DCM.M.pE;
        pC = DCM.M.pC;
        fprintf('Using existing priors\n')
    end
end
 
% initial states and equations of motion
%--------------------------------------------------------------------------
[x,f]    = spm_dcm_x_neural(pE,model);

% check for pre-specified priors
%--------------------------------------------------------------------------
hE       = 8;
hC       = 1/128;
try, hE  = DCM.M.hE; hC  = DCM.M.hC; end
 
% create DCM
%--------------------------------------------------------------------------
DCM.M.IS = 'spm_csd_mtf';
DCM.M.g  = 'spm_gx_erp';
DCM.M.f  = f;
DCM.M.x  = x;
DCM.M.n  = length(spm_vec(x));
DCM.M.pE = pE;
DCM.M.pC = pC;
DCM.M.hE = hE;
DCM.M.hC = hC;
DCM.M.m  = Ns;

% specify M.u - endogenous input (fluctuations) and intial states
%--------------------------------------------------------------------------
DCM.M.u  = sparse(Ns,1);

%-Feature selection using principal components (U) of lead-field
%==========================================================================
 
% Spatial modes
%--------------------------------------------------------------------------
try
    DCM.M.U = spm_dcm_eeg_channelmodes(DCM.M.dipfit,Nm);
end
 
% get data-features (in reduced eigenspace)
%==========================================================================
%if DATA
%    DCM  = spm_dcm_csd_data(DCM);
%end
 
% scale data features (to a variance of about 8)
%--------------------------------------------------------------------------
% ccf      = spm_csd2ccf(DCM.xY.y,DCM.xY.Hz);
% scale    = max(spm_vec(ccf));
% DCM.xY.y = spm_unvec(8*spm_vec(DCM.xY.y)/scale,DCM.xY.y);


% complete model specification and invert
%==========================================================================
Nm       = size(DCM.M.U,2);                    % number of spatial modes
DCM.M.l  = Nm;
DCM.M.Hz = DCM.xY.Hz;
DCM.M.dt = DCM.xY.dt;
 
% normalised precision
%--------------------------------------------------------------------------
DCM.xY.Q  = spm_dcm_csd_Q(DCM.xY.y);
DCM.xY.X0 = sparse(size(DCM.xY.Q,1),0);

DCM.M.pE.J = [0 0 0 0 0 0 0 0 1 0 0 0 0];
pE.J=DCM.M.pE.J;
DCM.M.pC.J = [0.0312 0 0 0 0 0 0.0312 0 0 0 0 0 0];
pC.J = DCM.M.pC.J;

load("params.mat", 'params');
% Use MOGA-informed priors
params_all = struct(...
    'R', [MOGA_params(1,nsim),params(2)],...
    'T', [params(3), params(4)],...
    'G', params(5),...
    'H', [params(6), params(7), params(8), params(9), params(10)],...
    'A', [params(11), params(12), params(13)],...
    'C', 0,...
    'D', MOGA_params(2,nsim),...
    'I', params(15),...
    'Lpos', [0; 0; 0],...
    'L', 1,...
    'J', [0 0 0 0 0 0 0 0 1 0 0 0 0],...
    'a', [params(16); params(17)],...
    'b', [params(18); params(19)],...
    'c', [params(20); params(21)],...
    'd', [params(22); params(23); params(24); params(25)],...
    'f', [params(26); params(27)]);
params_all.A=num2cell(params_all.A);
pE=params_all;
DCM.M.pE=pE;


params_2 = struct(...
    'R', [1,0],...
    'T', [0, 0],...
    'G', 0,...
    'H', [0, 0, 0, 0, 0],...
    'A', [0, 0, 0],...
    'C', 0,...
    'D', 1,...
    'I', 0,...
    'Lpos', [0; 0; 0],...
    'L', 64,...
    'J', [0.0312 0 0 0 0 0 0.0312 0 0 0 0 0 0],...
    'a', [0; 0],...
    'b', [0; 0],...
    'c', [0; 0],...
    'd', [0; 0; 0; 0],...
    'f', [0; 0]);
params_2.A=num2cell(params_2.A);
pC=params_2;
DCM.M.pC=pC;

tic;
%Variational Laplace: model inversion
%==========================================================================
[Qp,Cp,Eh,F] = spm_nlsi_GN(DCM.M,DCM.xU,DCM.xY);


% Data ID
%--------------------------------------------------------------------------
try
    try
        ID = spm_data_id(feval(DCM.M.FS,DCM.xY.y,DCM.M));
    catch
        ID = spm_data_id(feval(DCM.M.FS,DCM.xY.y));
    end
catch
    ID = spm_data_id(DCM.xY.y);
end
 
 
% Bayesian inference {threshold = prior} NB Prior on A,B and C = exp(0) = 1
%==========================================================================
warning('off','SPM:negativeVariance');
dp  = spm_vec(Qp) - spm_vec(pE);
Pp  = spm_unvec(1 - spm_Ncdf(0,abs(dp),diag(Cp)),Qp);
warning('on', 'SPM:negativeVariance');
 
 
% predictions (csd) and error (sensor space)
%--------------------------------------------------------------------------
Hc  = spm_csd_mtf(Qp,DCM.M,DCM.xU); % prediction
%Hc = log(Hc{1}); %log-scaled spectrum
Ec  = spm_unvec(spm_vec(DCM.xY.y) - spm_vec(Hc),Hc);     % prediction error
 
 
% predictions (source space - cf, a LFP from virtual electrode)
%--------------------------------------------------------------------------
M             = rmfield(DCM.M,'U'); 
M.dipfit.type = 'LFP';

M.U         = 1;
M.l         = Ns;
qp          = Qp;
qp.L        = ones(1,Ns);             % set virtual electrode gain to unity
qp.b        = qp.b - 32;              % and suppress non-specific and
qp.c        = qp.c - 32;              % specific channel noise

[Hs Hz dtf] = spm_csd_mtf(qp,M,DCM.xU);
%Hs = log(Hs{1});
% dtf= log(dtf{1});
[ccf pst]   = spm_csd2ccf(Hs,DCM.M.Hz);
[coh fsd]   = spm_csd2coh(Hs,DCM.M.Hz);
DCM.dtf     = dtf;
DCM.ccf     = ccf;
DCM.coh     = coh;
DCM.fsd     = fsd;
DCM.pst     = pst;
DCM.Hz      = Hz;

 
% store estimates in DCM
%--------------------------------------------------------------------------
DCM.Ep = Qp;                   % conditional expectation
DCM.Cp = Cp;                   % conditional covariance
DCM.Pp = Pp;                   % conditional probability
DCM.Hc = Hc;                   % conditional responses (y), channel space
DCM.Rc = Ec;                   % conditional residuals (y), channel space
DCM.Hs = Hs;                   % conditional responses (y), source space
DCM.Ce = exp(-Eh);             % ReML error covariance
DCM.F  = F;                    % Laplace log evidence
DCM.ID = ID;                   % data ID

%display(Qp.J);

DCM.runtime_dcm=toc;

% and save
%--------------------------------------------------------------------------
DCM.options.Nmodes = Nm;

% DCM.name = ['Grand_post-PL_LFP_MOGA_means_' num2str(i) '_' DCM.name];
%DCM.name = ['Grand_control_LFP_' num2str(i) '_' DCM.name];
%save(DCM.name, 'DCM', spm_get_defaults('mat.format'));

return

% NOTES: for population specific cross spectra
%--------------------------------------------------------------------------
M             = rmfield(DCM.M,'U'); 
M.dipfit.type = 'LFP';
M           = DCM.M;
M.U         = 1; 
M.l         = DCM.M.m;
qp          = DCM.Ep;
qp.L        = ones(1,M.l);              % set electrode gain to unity
qp.b        = qp.b - 32;                % and suppress non-specific and
qp.c        = qp.c - 32;                % specific channel noise

% specifying the j-th population in the i-th source
%--------------------------------------------------------------------------
i           = 1;
j           = 2;
qp.J{i}     = spm_zeros(qp.J{i});
qp.J{i}(j)  = 1;

[Hs Hz dtf] = spm_csd_mtf(qp,M,DCM.xU); % conditional cross spectra
% Hs = log(Hs{1});
% dtf= log(dtf{1});

[ccf pst]   = spm_csd2ccf(Hs,DCM.M.Hz); % conditional correlation functions
[coh fsd]   = spm_csd2coh(Hs,DCM.M.Hz); % conditional covariance

