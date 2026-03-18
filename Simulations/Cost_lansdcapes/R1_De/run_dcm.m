%% Run Standard DCM
paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
DCM = [];
DCM.A = {[1],[1],[1]};
DCM.B = {};
DCM.C = 0;

load('synthetic_psd_params_R1De.mat','model_spec_1', 'model_spec_2', 'params_1', 'params_2');
spec=[model_spec_1,model_spec_2];
freq=2:0.5:30;

for dat=1:2
DCM.xY.y{1} = spec(:,dat);% %[ put spectrum vector here] % data

DCM.xY.Hz = freq; % frequency from 1 to 20 Hz
DCM.xY.dt = 1;
options.Nmodes       =1;% number of spatial modes
%options.Tdcm         [- [start end] time window in ms

options.Fdcm         = DCM.xY.Hz([1 end]);%[start end] Frequency window in Hz
options.D            =1;%- time bin decimation       (usually 1 or 2)
options.spatial      ='LFP';%- 'ECD', 'LFP' or 'IMG'     (see spm_erp_L)
options.model        ='LFP';%- 'ERP', 'SEP', 'CMC', 'LFP', 'NMM' or 'MFM'

DCM.options = options;
% check options
%==========================================================================
drawnow
clear spm_erp_L
%name = sprintf('DCM_%s',date);
DCM.options.analysis  = 'CSD';

% Filename and options
%--------------------------------------------------------------------------
%try, DCM.name;                      catch, DCM.name = name;      end
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
% 
DCM.M.pE.J = [0 0 0 0 0 0 0 0 1 0 0 0 0];
pE.J=DCM.M.pE.J;
DCM.M.pC.J = [0.0312 0 0 0 0 0 0.0312 0 0 0 0 0 0];
pC.J = DCM.M.pC.J;

load("params.mat");
DCM.M.pE=pE;
DCM.M.pE.R(2)=params(2);
DCM.M.pE.T(1)=params(3);
DCM.M.pE.T(2)=params(4);
DCM.M.pE.G=params(5);
DCM.M.pE.H(1)=params(6);
DCM.M.pE.H(2)=params(7);
DCM.M.pE.H(3)=params(8);
DCM.M.pE.H(4)=params(9);
DCM.M.pE.H(5)=params(10);
DCM.M.pE.A{1}=params(11);
DCM.M.pE.A{2}=params(12);
DCM.M.pE.A{3}=params(13);
DCM.M.pE.I=params(15);
DCM.M.pE.a(1)=params(16);
DCM.M.pE.a(2)=params(17);
DCM.M.pE.b(1)=params(18);
DCM.M.pE.b(2)=params(19);
DCM.M.pE.c(1)=params(20);
DCM.M.pE.c(2)=params(21);
DCM.M.pE.d(1)=params(22);
DCM.M.pE.d(2)=params(23);
DCM.M.pE.d(3)=params(24);
DCM.M.pE.d(4)=params(25);
DCM.M.pE.f(1)=params(26);
DCM.M.pE.f(2)=params(27);


DCM.M.pC=pC;
DCM.M.pC.R(2)=0;
DCM.M.pC.T(1)=0;
DCM.M.pC.T(2)=0;
DCM.M.pC.G=0;
DCM.M.pC.H(1)=0;
DCM.M.pC.H(2)=0;
DCM.M.pC.H(3)=0;
DCM.M.pC.H(4)=0;
DCM.M.pC.H(5)=0;
DCM.M.pC.A{1}=0;
DCM.M.pC.A{2}=0;
DCM.M.pC.A{3}=0;
DCM.M.pC.I=0;
DCM.M.pC.a(1)=0;
DCM.M.pC.a(2)=0;
DCM.M.pC.b(1)=0;
DCM.M.pC.b(2)=0;
DCM.M.pC.c(1)=0;
DCM.M.pC.c(2)=0;
DCM.M.pC.d(1)=0;
DCM.M.pC.d(2)=0;
DCM.M.pC.d(3)=0;
DCM.M.pC.d(4)=0;
DCM.M.pC.f(1)=0;
DCM.M.pC.f(2)=0;



% Run inversion on simulated data

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
% and save
%--------------------------------------------------------------------------
DCM.options.Nmodes = Nm;

DCM_all{dat}=DCM;
end 

save('standard_DCM_synthetic_spec_fixed.mat', 'DCM_all');

%% Run DCM with wider variances

paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
DCM = [];
DCM.A = {[1],[1],[1]};
DCM.B = {};
DCM.C = 0;

load('synthetic_psd_params_R1De.mat','model_spec_1', 'model_spec_2', 'params_1', 'params_2');
spec=[model_spec_1,model_spec_2];
freq=2:0.5:30;

for dat=1:2
DCM.xY.y{1} = spec(:,dat);% %[ put spectrum vector here] % data

DCM.xY.Hz = freq; % frequency from 1 to 20 Hz
DCM.xY.dt = 1;
options.Nmodes       =1;% number of spatial modes
%options.Tdcm         [- [start end] time window in ms

options.Fdcm         = DCM.xY.Hz([1 end]);%[start end] Frequency window in Hz
options.D            =1;%- time bin decimation       (usually 1 or 2)
options.spatial      ='LFP';%- 'ECD', 'LFP' or 'IMG'     (see spm_erp_L)
options.model        ='LFP';%- 'ERP', 'SEP', 'CMC', 'LFP', 'NMM' or 'MFM'

DCM.options = options;
% check options
%==========================================================================
drawnow
clear spm_erp_L
%name = sprintf('DCM_%s',date);
DCM.options.analysis  = 'CSD';

% Filename and options
%--------------------------------------------------------------------------
%try, DCM.name;                      catch, DCM.name = name;      end
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
% 
DCM.M.pE.J = [0 0 0 0 0 0 0 0 1 0 0 0 0];
pE.J=DCM.M.pE.J;
DCM.M.pC.J = [0.0312 0 0 0 0 0 0.0312 0 0 0 0 0 0];
pC.J = DCM.M.pC.J;

load("params.mat");
DCM.M.pE=pE;
DCM.M.pE.R(2)=params(2);
DCM.M.pE.T(1)=params(3);
DCM.M.pE.T(2)=params(4);
DCM.M.pE.G=params(5);
DCM.M.pE.H(1)=params(6);
DCM.M.pE.H(2)=params(7);
DCM.M.pE.H(3)=params(8);
DCM.M.pE.H(4)=params(9);
DCM.M.pE.H(5)=params(10);
DCM.M.pE.A{1}=params(11);
DCM.M.pE.A{2}=params(12);
DCM.M.pE.A{3}=params(13);
DCM.M.pE.I=params(15);
DCM.M.pE.a(1)=params(16);
DCM.M.pE.a(2)=params(17);
DCM.M.pE.b(1)=params(18);
DCM.M.pE.b(2)=params(19);
DCM.M.pE.c(1)=params(20);
DCM.M.pE.c(2)=params(21);
DCM.M.pE.d(1)=params(22);
DCM.M.pE.d(2)=params(23);
DCM.M.pE.d(3)=params(24);
DCM.M.pE.d(4)=params(25);
DCM.M.pE.f(1)=params(26);
DCM.M.pE.f(2)=params(27);

% Var 1
DCM.M.pC=pC;
DCM.M.pC.R(1)=1;
DCM.M.pC.R(2)=0;
DCM.M.pC.T(1)=0;
DCM.M.pC.T(2)=0;
DCM.M.pC.G=0;
DCM.M.pC.H(1)=0;
DCM.M.pC.H(2)=0;
DCM.M.pC.H(3)=0;
DCM.M.pC.H(4)=0;
DCM.M.pC.H(5)=0;
DCM.M.pC.A{1}=0;
DCM.M.pC.A{2}=0;
DCM.M.pC.A{3}=0;
DCM.M.pC.D=1;
DCM.M.pC.I=0;
DCM.M.pC.a(1)=0;
DCM.M.pC.a(2)=0;
DCM.M.pC.b(1)=0;
DCM.M.pC.b(2)=0;
DCM.M.pC.c(1)=0;
DCM.M.pC.c(2)=0;
DCM.M.pC.d(1)=0;
DCM.M.pC.d(2)=0;
DCM.M.pC.d(3)=0;
DCM.M.pC.d(4)=0;
DCM.M.pC.f(1)=0;
DCM.M.pC.f(2)=0;

%var 2
% DCM.M.pC=pC;
% DCM.M.pC.R(1)=10;
% DCM.M.pC.R(2)=0;
% DCM.M.pC.T(1)=0;
% DCM.M.pC.T(2)=0;
% DCM.M.pC.G=0;
% DCM.M.pC.H(1)=0;
% DCM.M.pC.H(2)=0;
% DCM.M.pC.H(3)=0;
% DCM.M.pC.H(4)=0;
% DCM.M.pC.H(5)=0;
% DCM.M.pC.A{1}=0;
% DCM.M.pC.A{2}=0;
% DCM.M.pC.A{3}=0;
% DCM.M.pC.D=10;
% DCM.M.pC.I=0;
% DCM.M.pC.a(1)=0;
% DCM.M.pC.a(2)=0;
% DCM.M.pC.b(1)=0;
% DCM.M.pC.b(2)=0;
% DCM.M.pC.c(1)=0;
% DCM.M.pC.c(2)=0;
% DCM.M.pC.d(1)=0;
% DCM.M.pC.d(2)=0;
% DCM.M.pC.d(3)=0;
% DCM.M.pC.d(4)=0;
% DCM.M.pC.f(1)=0;
% DCM.M.pC.f(2)=0;

%var 3
% DCM.M.pC=pC;
% DCM.M.pC.R(1)=DCM.M.pC.R(1)*10;
% DCM.M.pC.R(2)=0;
% DCM.M.pC.T(1)=0;
% DCM.M.pC.T(2)=0;
% DCM.M.pC.G=0;
% DCM.M.pC.H(1)=0;
% DCM.M.pC.H(2)=0;
% DCM.M.pC.H(3)=0;
% DCM.M.pC.H(4)=0;
% DCM.M.pC.H(5)=0;
% DCM.M.pC.A{1}=0;
% DCM.M.pC.A{2}=0;
% DCM.M.pC.A{3}=0;
% DCM.M.pC.D=DCM.M.pC.D*10;
% DCM.M.pC.I=0;
% DCM.M.pC.a(1)=0;
% DCM.M.pC.a(2)=0;
% DCM.M.pC.b(1)=0;
% DCM.M.pC.b(2)=0;
% DCM.M.pC.c(1)=0;
% DCM.M.pC.c(2)=0;
% DCM.M.pC.d(1)=0;
% DCM.M.pC.d(2)=0;
% DCM.M.pC.d(3)=0;
% DCM.M.pC.d(4)=0;
% DCM.M.pC.f(1)=0;
% DCM.M.pC.f(2)=0;


% Run inversion on simulated data

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
% and save
%--------------------------------------------------------------------------
DCM.options.Nmodes = Nm;

DCM_all{dat}=DCM;
end 

save('wider1Var_DCM_synthetic_spec_fixed.mat', 'DCM_all');

%% Run DCM with different priors/bounds spanning the param bounds

paramsvec = {'R1','R2','Te','Ti','He', 'G1', 'G2', 'G3', 'G4', 'G5','A1','A2','A3','De','Di','a1','a2','b1','b2','c1','c2','d1','d2','d3','d4','f1','f2'};
DCM = [];
DCM.A = {[1],[1],[1]};
DCM.B = {};
DCM.C = 0;

load('synthetic_psd_params_R1De.mat','model_spec_1', 'model_spec_2', 'params_1', 'params_2');
spec=[model_spec_1,model_spec_2];
freq=2:0.5:30;

for dat=1:2
DCM.xY.y{1} = spec(:,dat);% %[ put spectrum vector here] % data

DCM.xY.Hz = freq; % frequency from 1 to 20 Hz
DCM.xY.dt = 1;
options.Nmodes       =1;% number of spatial modes
%options.Tdcm         [- [start end] time window in ms

options.Fdcm         = DCM.xY.Hz([1 end]);%[start end] Frequency window in Hz
options.D            =1;%- time bin decimation       (usually 1 or 2)
options.spatial      ='LFP';%- 'ECD', 'LFP' or 'IMG'     (see spm_erp_L)
options.model        ='LFP';%- 'ERP', 'SEP', 'CMC', 'LFP', 'NMM' or 'MFM'

DCM.options = options;
% check options
%==========================================================================
drawnow
clear spm_erp_L
%name = sprintf('DCM_%s',date);
DCM.options.analysis  = 'CSD';

% Filename and options
%--------------------------------------------------------------------------
%try, DCM.name;                      catch, DCM.name = name;      end
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
% 
DCM.M.pE.J = [0 0 0 0 0 0 0 0 1 0 0 0 0];
pE.J=DCM.M.pE.J;
DCM.M.pC.J = [0.0312 0 0 0 0 0 0.0312 0 0 0 0 0 0];
pC.J = DCM.M.pC.J;

load("params.mat");

lb = [-1.4142 -1.4142 -1.4142 -1.4142 -1.1068 -1.0046 -1.3208 -1.0046 -1.0046 -1.0846 -3.2533 -2.8284 -3.2769 -1.0000 -0.7071 -0.3536 -0.5851 -0.3774 -0.3536 -0.3774 -0.3536 -0.3634 -0.3536 -0.3536 -0.4002 -0.5000 -0.6625]; 
ub = [1.4142 1.4142 1.4142 1.4142 1 1 1 1.0509 1.0006 1.1017 2.8284 2.9713 2.8284 1 0.7768 0.3536 0.3536 0.3536 0.3562 0.3536 0.3562 0.3536 0.3712 0.4410 0.3536 0.5000 0.5000];

mid_point = (lb + ub).'/2;

z = norminv(0.995);   % 99% two-sided
mu = (ub - lb).'/(2*z);
var = mu.^2;

DCM.M.pE=pE;
DCM.M.pE.R(1)=mid_point(1);
DCM.M.pE.R(2)=params(2);
DCM.M.pE.T(1)=params(3);
DCM.M.pE.T(2)=params(4);
DCM.M.pE.G=params(5);
DCM.M.pE.H(1)=params(6);
DCM.M.pE.H(2)=params(7);
DCM.M.pE.H(3)=params(8);
DCM.M.pE.H(4)=params(9);
DCM.M.pE.H(5)=params(10);
DCM.M.pE.A{1}=params(11);
DCM.M.pE.A{2}=params(12);
DCM.M.pE.A{3}=params(13);
DCM.M.pE.D=mid_point(14);
DCM.M.pE.I=params(15);
DCM.M.pE.a(1)=params(16);
DCM.M.pE.a(2)=params(17);
DCM.M.pE.b(1)=params(18);
DCM.M.pE.b(2)=params(19);
DCM.M.pE.c(1)=params(20);
DCM.M.pE.c(2)=params(21);
DCM.M.pE.d(1)=params(22);
DCM.M.pE.d(2)=params(23);
DCM.M.pE.d(3)=params(24);
DCM.M.pE.d(4)=params(25);
DCM.M.pE.f(1)=params(26);
DCM.M.pE.f(2)=params(27);


DCM.M.pC=pC;
DCM.M.pC.R(1)=var(1);
DCM.M.pC.R(2)=0;
DCM.M.pC.T(1)=0;
DCM.M.pC.T(2)=0;
DCM.M.pC.G=0;
DCM.M.pC.H(1)=0;
DCM.M.pC.H(2)=0;
DCM.M.pC.H(3)=0;
DCM.M.pC.H(4)=0;
DCM.M.pC.H(5)=0;
DCM.M.pC.A{1}=0;
DCM.M.pC.A{2}=0;
DCM.M.pC.A{3}=0;
DCM.M.pC.D=var(14);
DCM.M.pC.I=0;
DCM.M.pC.a(1)=0;
DCM.M.pC.a(2)=0;
DCM.M.pC.b(1)=0;
DCM.M.pC.b(2)=0;
DCM.M.pC.c(1)=0;
DCM.M.pC.c(2)=0;
DCM.M.pC.d(1)=0;
DCM.M.pC.d(2)=0;
DCM.M.pC.d(3)=0;
DCM.M.pC.d(4)=0;
DCM.M.pC.f(1)=0;
DCM.M.pC.f(2)=0;


% Run inversion on simulated data

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
% and save
%--------------------------------------------------------------------------
DCM.options.Nmodes = Nm;

DCM_all{dat}=DCM;
end 

save('MidBoundsPriors_DCM_synthetic_spec_fixed.mat', 'DCM_all');

