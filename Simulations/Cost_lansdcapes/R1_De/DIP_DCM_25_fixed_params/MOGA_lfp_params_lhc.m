
R1 = 0;
D=0;


% Create lower bounds (lb) and upper bounds (ub)

lb = [-1.4142 -1]; 
ub = [1.4142 1];


% Parameter ranges of priors leading to good fits using DCM. 
paramsvec=[R1;D];

p1 = lhsdesign_scale_dom(npop, npars_est, [lb(paramstoest);ub(paramstoest)]);

