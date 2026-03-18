
R1 = 0;
I=0;


% Create lower bounds (lb) and upper bounds (ub)

lb = [-1.4142 -0.7071]; 
ub = [1.4142 0.7768];

% Parameter ranges of priors leading to good fits using DCM. 
paramsvec=[R1;I];

p1 = lhsdesign_scale_dom(npop, npars_est, [lb(paramstoest);ub(paramstoest)]);

